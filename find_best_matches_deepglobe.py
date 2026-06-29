import os
import glob
import random
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset

# ---------------------------------------------------------
# 1. Custom Dataset for DeepGlobe
# ---------------------------------------------------------
class DeepGlobeDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = sorted(glob.glob(os.path.join(data_dir, "*_sat.jpg")))
        self.masks = [img_path.replace("_sat.jpg", "_mask.png") for img_path in self.images]

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
        
        return image, mask, img_path

# ---------------------------------------------------------
# 2. Initialize Architecture & IoU
# ---------------------------------------------------------
def get_model():
    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights=None,       
        in_channels=3,              
        activation=None
    )
    return model

def calculate_iou(pred, target):
    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    if np.sum(union) == 0:
        return 0.0
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# ---------------------------------------------------------
# 3. Main Scanning Function
# ---------------------------------------------------------
def find_best_matches_deepglobe(model_path="deepglobe_finetuned_model.pth", data_dir="archive/train", num_images_to_scan=10000, top_k=50):
    print("\n========== CHERRY-PICKING THE BEST MATCHES (DEEPGLOBE) ==========")
    print(f"[*] Scanning {num_images_to_scan} random DeepGlobe images...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Device: {device}")
    
    model = get_model()
    if not os.path.exists(model_path):
        print(f"\n[!] ERROR: Cannot find {model_path}!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() 
    
    dataset = DeepGlobeDataset(data_dir)
    dataset_length = len(dataset)
    
    if dataset_length == 0:
        print("[!] ERROR: No DeepGlobe images found in", data_dir)
        return
        
    results = []
    indices_to_scan = random.sample(range(dataset_length), min(num_images_to_scan, dataset_length))
    
    print("[*] Processing images...")
    for count, idx in enumerate(indices_to_scan):
        image, mask_true, img_path = dataset[idx]
        image = image.float().unsqueeze(0) 
        
        # Filter completely blank masks
        if mask_true.sum() < 50:
            continue
        
        # DeepGlobe is already RGB, so no channel slicing needed.
        # Just normalize the batch min/max
        batch_min = image.amin(dim=(1, 2, 3), keepdim=True)
        batch_max = image.amax(dim=(1, 2, 3), keepdim=True)
        image = (image - batch_min) / (batch_max - batch_min + 1e-8)
        
        image = image.to(device)
        
        with torch.no_grad():
            logits = model(image) 
            prediction = torch.sigmoid(logits) 
            
        prediction = prediction.squeeze().cpu().numpy()
        mask_true = mask_true.squeeze().cpu().numpy()
        
        # Standard threshold for thick DeepGlobe highways
        prediction_thresholded = (prediction > 0.25).astype(float)
        
        iou = calculate_iou(prediction_thresholded, mask_true)
        results.append({
            'idx': idx,
            'file': os.path.basename(img_path),
            'iou': iou,
            'img_display': image.squeeze().permute(1, 2, 0).cpu().numpy(),
            'mask_true': mask_true,
            'prediction': prediction_thresholded
        })
        
        print(f"    Scanned {count+1}/{num_images_to_scan} (IoU: {iou:.4f})", end='\r')
        
    print("\n\n[*] Sorting to find the absolute best matches...")
    results.sort(key=lambda x: x['iou'], reverse=True)
    
    os.makedirs("Documentation/best_results", exist_ok=True)
    
    for rank in range(min(top_k, len(results))):
        best = results[rank]
        print(f"[+] Rank #{rank+1} -> File {best['file']} (IoU Score: {best['iou']:.4f})")
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        img_disp = best['img_display']
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())
        
        axs[0].imshow(img_disp)
        axs[0].set_title(f"DeepGlobe: {best['file']}")
        axs[0].axis('off')
        
        axs[1].imshow(best['mask_true'], cmap='gray')
        axs[1].set_title("Ground Truth")
        axs[1].axis('off')
        
        axs[2].imshow(best['prediction'], cmap='magma')
        axs[2].set_title(f"Model Prediction (IoU: {best['iou']:.4f})")
        axs[2].axis('off')
        
        plt.tight_layout()
        save_path = f"Documentation/best_results/deepglobe_best_match_rank_{rank+1}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"    -> Saved beautiful plot to {save_path}")

if __name__ == "__main__":
    find_best_matches_deepglobe()
