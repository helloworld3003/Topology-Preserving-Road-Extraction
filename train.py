import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import torchvision.models as models
import segmentation_models_pytorch as smp
from torchgeo.datasets import SpaceNet5

# ---------------------------------------------------------
# 1. Architecture Initialization
# ---------------------------------------------------------
def get_model():
    """
    Initializes a UNet++ model with a ResNet50 backbone.
    """
    model = smp.UnetPlusPlus(
        encoder_name="resnet50", 
        encoder_weights="imagenet", 
        in_channels=3,              
        classes=1,                  
        # Important: Remove 'sigmoid' activation here because BCEWithLogitsLoss expects raw logits!
        # This fixes numerical instability early in training.
        activation=None        
    )
    return model

# ---------------------------------------------------------
# 2. Advanced Loss Functions (CVPR 2021 + CVPR 2018)
# ---------------------------------------------------------
class VGGTopologyLoss(nn.Module):
    """
    CVPR 2018: Beyond the Pixel-Wise Loss for Topology-Aware Delineation.
    Extracts deep features from a frozen VGG-19 network to enforce topological structural similarity.
    """
    def __init__(self, device):
        super(VGGTopologyLoss, self).__init__()
        # Load VGG19 and freeze it
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
            
        # Extract specific layers according to paper: relu(conv1_2), relu(conv2_2), relu(conv3_4)
        # VGG19 feature indices: conv1_2 is at 3, conv2_2 is at 8, conv3_4 is at 17
        self.slice1 = vgg[:4].to(device)
        self.slice2 = vgg[4:9].to(device)
        self.slice3 = vgg[9:18].to(device)
        self.device = device
        
    def forward(self, y_pred_sigmoid, y_true):
        # VGG expects 3-channel input. Our masks are 1-channel.
        # We repeat the masks across 3 channels to feed them into VGG.
        pred_3c = y_pred_sigmoid.repeat(1, 3, 1, 1)
        true_3c = y_true.repeat(1, 3, 1, 1)
        
        # Get features for predicted mask
        pred_f1 = self.slice1(pred_3c)
        pred_f2 = self.slice2(pred_f1)
        pred_f3 = self.slice3(pred_f2)
        
        # Get features for ground truth mask
        true_f1 = self.slice1(true_3c)
        true_f2 = self.slice2(true_f1)
        true_f3 = self.slice3(true_f2)
        
        # Calculate L2 (MSE) distance between features
        loss_top = F.mse_loss(pred_f1, true_f1) + \
                   F.mse_loss(pred_f2, true_f2) + \
                   F.mse_loss(pred_f3, true_f3)
                   
        return loss_top


class SoftClDiceLoss(nn.Module):
    def __init__(self, iters=10, alpha=0.45):
        super(SoftClDiceLoss, self).__init__()
        self.iters = iters 
        self.alpha = alpha 

    def soft_skeletonize(self, x):
        S = torch.zeros_like(x)
        I = x
        for _ in range(self.iters):
            I_min = F.max_pool2d(-I, kernel_size=3, stride=1, padding=1)
            I_min = -I_min
            I_max = F.max_pool2d(I_min, kernel_size=3, stride=1, padding=1)
            diff = F.relu(I - I_max)
            S = S + (1 - S) * diff
            I = I_min
        return S

    def soft_dice(self, y_pred, y_true):
        smooth = 1e-5
        intersection = torch.sum(y_pred * y_true)
        return (2. * intersection + smooth) / (torch.sum(y_pred) + torch.sum(y_true) + smooth)

    def forward(self, y_pred, y_true):
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        
        smooth = 1e-5
        t_prec = (torch.sum(skel_pred * y_true) + smooth) / (torch.sum(skel_pred) + smooth)
        t_sens = (torch.sum(skel_true * y_pred) + smooth) / (torch.sum(skel_true) + smooth)
        
        cl_dice = 2.0 * (t_prec * t_sens) / (t_prec + t_sens)
        dice = self.soft_dice(y_pred, y_true)
        
        loss = (1.0 - self.alpha) * (1.0 - dice) + self.alpha * (1.0 - cl_dice)
        return loss

# ---------------------------------------------------------
# 3. Training Loop Integration
# ---------------------------------------------------------
def train_loop(aois=[8], batch_size=2, epochs=50, load_weights=None, save_weights="model_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Training Pipeline...")
    print(f"[*] Target Device: {device}")
    
    if device.type == "cpu":
        print("[!] WARNING: Running on CPU. Training will be slow.")
    else:
        print("[+] NVIDIA GPU Detected! AMP (Mixed Precision) Enabled.")
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spacenet_data")
    dataset = SpaceNet5(
        root=data_dir, 
        split="train", 
        aois=aois,
        download=False,
        checksum=False
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = get_model().to(device)
    
    if load_weights and os.path.exists(load_weights):
        print(f"[*] Loading pre-trained weights from: {load_weights}")
        model.load_state_dict(torch.load(load_weights, map_location=device, weights_only=True))
    
    # We switch to AdamW, which handles weight decay (L2 regularization) mathematically better than Adam
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Cosine Annealing Learning Rate Scheduler: Smoothly drops LR down to 1e-6 as epochs progress
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Hybrid Loss Components
    bce_loss_fn = nn.BCEWithLogitsLoss()
    cldice_loss_fn = SoftClDiceLoss(iters=10, alpha=0.45)
    
    # CVPR 2018 VGG Topology Loss (Only load it if on GPU to save CPU memory, as VGG is huge)
    vgg_topology_loss_fn = None
    if device.type == "cuda":
        print("[*] Loading VGG-19 for CVPR 2018 Perceptual Topology Loss...")
        vgg_topology_loss_fn = VGGTopologyLoss(device=device)

    # AMP Scaler for lightning-fast GPU training
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, sample in enumerate(dataloader):
            images = sample['image'].float()
            masks = sample['mask'].float()
            
            # Extract RGB
            images = images[:, [4, 2, 1], :, :]
            
            # Normalization
            batch_min = images.amin(dim=(1, 2, 3), keepdim=True)
            batch_max = images.amax(dim=(1, 2, 3), keepdim=True)
            images = (images - batch_min) / (batch_max - batch_min + 1e-8)
            
            # Crop to 320x320
            images = TF.center_crop(images, [320, 320])
            masks = TF.center_crop(masks, [320, 320])
            
            # --- DATA AUGMENTATION (Anti-Overfitting) ---
            # Random 50% chance to flip horizontally
            if random.random() > 0.5:
                images = TF.hflip(images)
                masks = TF.hflip(masks)
            # Random 50% chance to flip vertically
            if random.random() > 0.5:
                images = TF.vflip(images)
                masks = TF.vflip(masks)
                
            masks = masks.unsqueeze(1)
            
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # --- FORWARD PASS (WITH AMP) ---
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
                
                # BCE needs raw logits (pre-sigmoid) for mathematical stability
                loss_bce = bce_loss_fn(logits, masks)
                
                # clDice and VGG need actual probabilities [0.0, 1.0]
                probs = torch.sigmoid(logits)
                loss_cldice = cldice_loss_fn(probs, masks)
                
                # Total baseline loss
                total_loss = loss_bce + loss_cldice
                
                # Add VGG Perceptual Loss if on GPU
                if vgg_topology_loss_fn is not None:
                    loss_vgg = vgg_topology_loss_fn(probs, masks)
                    # Weight the VGG loss at 0.1 so it gently guides topology without overpowering BCE
                    total_loss = total_loss + (0.1 * loss_vgg)

            # --- BACKWARD PASS (WITH AMP) ---
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += total_loss.item()
            print(f"    Batch {batch_idx+1}/{len(dataloader)} | Total Loss: {total_loss.item():.4f}", end='\r', flush=True)
                
        avg_loss = epoch_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        # Step the scheduler to smoothly lower the learning rate
        scheduler.step()
        
        torch.save(model.state_dict(), save_weights)
        print(f"    [+] Model checkpoint saved to {save_weights}")
        
    print("\n[*] Training completely finished!")

if __name__ == "__main__":
    print("\n========== PHASE 1: TRAINING ON MUMBAI ==========")
    train_loop(
        aois=[8], 
        # When you switch to your RTX 3050, AMP cuts memory in half, so bump this to 4 or 8!
        batch_size=4, 
        epochs=50,
        # load_weights="mumbai_road_model.pth", 
        save_weights="mumbai_road_model.pth"
    )
