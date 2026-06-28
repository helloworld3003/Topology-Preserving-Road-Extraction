import os
import glob
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import torchvision.models as models
import segmentation_models_pytorch as smp

# ---------------------------------------------------------
# 1. Custom Dataset for DeepGlobe (RGB .jpg and .png masks)
# ---------------------------------------------------------
class DeepGlobeDataset(Dataset):
    def __init__(self, data_dir):
        """
        Assumes data_dir contains both the images and masks with the naming convention:
        [id]_sat.jpg and [id]_mask.png
        """
        self.data_dir = data_dir
        
        # Grab all *_sat.jpg files in the directory
        self.images = sorted(glob.glob(os.path.join(data_dir, "*_sat.jpg")))
        
        # Build the mask paths by replacing _sat.jpg with _mask.png
        self.masks = [img_path.replace("_sat.jpg", "_mask.png") for img_path in self.images]
        
        if len(self.images) == 0 and os.path.exists(data_dir):
            print(f"[!] Warning: No *_sat.jpg files found in {data_dir}")
        elif len(self.images) > 0:
            # Just verify the first mask exists to ensure the naming convention holds
            if not os.path.exists(self.masks[0]):
                print(f"[!] Warning: Mask not found for {self.images[0]}. Expected {self.masks[0]}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        image = TF.center_crop(image, [320, 320])
        mask = TF.center_crop(mask, [320, 320])
        
        return image, mask

# ---------------------------------------------------------
# 2. Architecture Initialization
# ---------------------------------------------------------
def get_model():
    model = smp.UnetPlusPlus(
        encoder_name="resnet50", 
        encoder_weights="imagenet", 
        in_channels=3,              
        classes=1,                  
        # Removed 'sigmoid' so BCELoss gets raw logits!
        activation=None        
    )
    return model

# ---------------------------------------------------------
# 3. Advanced Loss Functions (CVPR 2021 + CVPR 2018)
# ---------------------------------------------------------
class VGGTopologyLoss(nn.Module):
    def __init__(self, device):
        super(VGGTopologyLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.slice1 = vgg[:4].to(device)
        self.slice2 = vgg[4:9].to(device)
        self.slice3 = vgg[9:18].to(device)
        self.device = device
        
    def forward(self, y_pred_sigmoid, y_true):
        pred_3c = y_pred_sigmoid.repeat(1, 3, 1, 1)
        true_3c = y_true.repeat(1, 3, 1, 1)
        
        pred_f1 = self.slice1(pred_3c)
        pred_f2 = self.slice2(pred_f1)
        pred_f3 = self.slice3(pred_f2)
        
        true_f1 = self.slice1(true_3c)
        true_f2 = self.slice2(true_f1)
        true_f3 = self.slice3(true_f2)
        
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
# 4. Training Loop Integration
# ---------------------------------------------------------
def train_loop(data_dir, batch_size=2, epochs=50, load_weights=None, save_weights="model_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing DeepGlobe Training Pipeline...")
    print(f"[*] Target Device: {device}")
    
    if device.type == "cpu":
        print("[!] WARNING: Running on CPU. Training will be slow.")
    else:
        print("[+] NVIDIA GPU Detected! AMP (Mixed Precision) Enabled.")
    
    print(f"[*] Loading DeepGlobe from: {data_dir}")
    dataset = DeepGlobeDataset(data_dir)
    
    if len(dataset) == 0:
        print("[!] Error: Dataset is empty. Check your DeepGlobe extracted folders!")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = get_model().to(device)
    
    if load_weights and os.path.exists(load_weights):
        print(f"[*] Loading pre-trained weights from: {load_weights}")
        model.load_state_dict(torch.load(load_weights, map_location=device, weights_only=True))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    bce_loss_fn = nn.BCEWithLogitsLoss()
    cldice_loss_fn = SoftClDiceLoss(iters=10, alpha=0.45)
    
    vgg_topology_loss_fn = None
    if device.type == "cuda":
        print("[*] Loading VGG-19 for CVPR 2018 Perceptual Topology Loss...")
        vgg_topology_loss_fn = VGGTopologyLoss(device=device)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            # Normalization
            batch_min = images.amin(dim=(1, 2, 3), keepdim=True)
            batch_max = images.amax(dim=(1, 2, 3), keepdim=True)
            images = (images - batch_min) / (batch_max - batch_min + 1e-8)
            
            # --- DATA AUGMENTATION (Anti-Overfitting) ---
            if random.random() > 0.5:
                images = TF.hflip(images)
                masks = TF.hflip(masks)
            if random.random() > 0.5:
                images = TF.vflip(images)
                masks = TF.vflip(masks)
            
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(images)
                
                loss_bce = bce_loss_fn(logits, masks)
                probs = torch.sigmoid(logits)
                loss_cldice = cldice_loss_fn(probs, masks)
                
                total_loss = loss_bce + loss_cldice
                
                if vgg_topology_loss_fn is not None:
                    loss_vgg = vgg_topology_loss_fn(probs, masks)
                    total_loss = total_loss + (0.1 * loss_vgg)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += total_loss.item()
            print(f"    Batch {batch_idx+1}/{len(dataloader)} | Total Loss: {total_loss.item():.4f}", end='\r')
                
        avg_loss = epoch_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        
        scheduler.step()
        torch.save(model.state_dict(), save_weights)
        print(f"    [+] Model checkpoint saved to {save_weights}")
        
    print("\n[*] Training completely finished!")

if __name__ == "__main__":
    # The user extracted DeepGlobe into a 'train' folder and then moved it into 'archive'
    DEEPGLOBE_TRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive", "train")
    
    if not os.path.exists(DEEPGLOBE_TRAIN_DIR):
        print(f"[!] Error: The folder {DEEPGLOBE_TRAIN_DIR} does not exist.")
    else:
        print("\n========== DEEPGLOBE TRANSFER LEARNING ==========")
        train_loop(
            data_dir=DEEPGLOBE_TRAIN_DIR,
            batch_size=2, # Bump to 8 on GPU!
            epochs=50,
            # load_weights="mumbai_road_model.pth",      
            save_weights="deepglobe_road_model.pth"    
        )
