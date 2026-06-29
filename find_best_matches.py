import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from torchgeo.datasets import SpaceNet5

# ---------------------------------------------------------
# 1. Initialize Identical Model Architecture
# ---------------------------------------------------------
def get_model():
    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights=None,       
        in_channels=3,              
        activation=None
    )
    return model

# ---------------------------------------------------------
# 2. IoU Calculation Logic
# ---------------------------------------------------------
def calculate_iou(pred, target):
    # Both pred and target should be binary [0, 1] arrays
    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    if np.sum(union) == 0:
        return 0.0 # Avoid division by zero if both are completely empty
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# ---------------------------------------------------------
# 3. Main Scanning Function
# ---------------------------------------------------------
def find_best_matches(model_path="mumbai_finetuned_model.pth", num_images_to_scan=100, top_k=3):
    print("\n========== CHERRY-PICKING THE BEST MATCHES ==========")
    print(f"[*] Scanning {num_images_to_scan} random SpaceNet 5 images to find the highest accuracy predictions...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Device: {device}")
    
    model = get_model()
    if not os.path.exists(model_path):
        print(f"\n[!] ERROR: Cannot find {model_path}!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() 
    
    dataset = SpaceNet5(root="spacenet_data", aois=[8], split="train", download=False)
    dataset_length = len(dataset)
    
    results = []
    
    # We will pick random indices to scan
    indices_to_scan = random.sample(range(dataset_length), min(num_images_to_scan, dataset_length))
    
    print("[*] Processing images...")
    for count, idx in enumerate(indices_to_scan):
        sample = dataset[idx]
        image = sample['image'].float().unsqueeze(0) 
        mask_true = sample['mask'].float()           
        
        # Only process images that actually have roads in the ground truth
        # (Otherwise the IoU will be 0/0 and it wastes a slot)
        if mask_true.sum() < 50:
            continue
            
        image_rgb = image[:, [4, 2, 1], :, :]
        image_rgb = TF.center_crop(image_rgb, [320, 320])
        mask_true = TF.center_crop(mask_true, [320, 320])
        
        batch_min = image_rgb.amin(dim=(1, 2, 3), keepdim=True)
        batch_max = image_rgb.amax(dim=(1, 2, 3), keepdim=True)
        image_rgb = (image_rgb - batch_min) / (batch_max - batch_min + 1e-8)
        
        image_rgb = image_rgb.to(device)
        
        with torch.no_grad():
            logits = model(image_rgb) 
            prediction = torch.sigmoid(logits) 
            
        prediction = prediction.squeeze().cpu().numpy()
        mask_true = mask_true.squeeze().cpu().numpy()
        
        # We use 0.3 threshold as discussed earlier for faint roads!
        prediction_thresholded = (prediction > 0.3).astype(float)
        
        iou = calculate_iou(prediction_thresholded, mask_true)
        results.append({
            'idx': idx,
            'iou': iou,
            'img_display': image_rgb.squeeze().permute(1, 2, 0).cpu().numpy(),
            'mask_true': mask_true,
            'prediction': prediction_thresholded
        })
        
        print(f"    Scanned {count+1}/{num_images_to_scan} (IoU: {iou:.4f})", end='\r')
        
    print("\n\n[*] Sorting to find the absolute best matches...")
    results.sort(key=lambda x: x['iou'], reverse=True)
    
    os.makedirs("Documentation", exist_ok=True)
    
    for rank in range(min(top_k, len(results))):
        best = results[rank]
        print(f"[+] Rank #{rank+1} -> Image #{best['idx']} (IoU Score: {best['iou']:.4f})")
        
        # Plot and save
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        img_disp = best['img_display']
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())
        
        axs[0].imshow(img_disp)
        axs[0].set_title(f"Rank {rank+1}: SpaceNet Image #{best['idx']}")
        axs[0].axis('off')
        
        axs[1].imshow(best['mask_true'], cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[1].axis('off')
        
        axs[2].imshow(best['prediction'], cmap='magma')
        axs[2].set_title(f"Model Prediction (IoU: {best['iou']:.4f})")
        axs[2].axis('off')
        
        plt.tight_layout()
        save_path = f"Documentation/best_results/best_match_rank_{rank+1}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"    -> Saved beautiful plot to {save_path}")

if __name__ == "__main__":
    # Scans 200 random images and saves the top 5 absolute best matches for the presentation!
    find_best_matches(model_path="mumbai_finetuned_model.pth", num_images_to_scan=2000, top_k=20)
