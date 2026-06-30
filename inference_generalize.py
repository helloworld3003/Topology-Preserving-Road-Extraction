import os
import random
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from torchgeo.datasets import SpaceNet5
from PIL import Image

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
# 2. Main Inference Function
# ---------------------------------------------------------
def test_generalization(model_path="mumbai_finetuned_model.pth", show_plot=False, image_path=None):
    print("\n========== CROSS-DATASET GENERALIZATION TEST ==========")
    print(f"[*] Testing the model on unseen satellite data!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Device: {device}")
    
    # 1. Load Model
    model = get_model()
    if not os.path.exists(model_path):
        print(f"\n[!] ERROR: Cannot find {model_path}!")
        return None
        
    print(f"[*] Loading Brain Data from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() 
    
    # 2. Load the Image
    if image_path is None:
        if "deepglobe" in model_path.lower():
            print("[*] Fetching a random test image from DeepGlobe (archive/train)...")
            import glob
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive", "train")
            all_images = sorted(glob.glob(os.path.join(data_dir, "*_sat.jpg")))
            original_tif_path = random.choice(all_images)
            
            from skimage import io
            img_np = io.imread(original_tif_path)
            image_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float()
        else:
            print("[*] Fetching a random unseen test image from SpaceNet 5 (Mumbai)...")
            dataset = SpaceNet5(root="spacenet_data", aois=[8], split="train", download=False)
            
            random_idx = random.randint(0, len(dataset) - 1)
            sample = dataset[random_idx] 
            original_tif_path = dataset.images[random_idx]
            image_tensor = sample['image'].float()
    else:
        print(f"[*] Loading custom image from {image_path}...")
        original_tif_path = image_path
        # Load image via skimage or PIL
        from skimage import io
        img_np = io.imread(image_path)
        # Convert to tensor (C, H, W)
        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=-1)
        image_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float()

    # Extract chip ID from filename for cleaner logging
    import re
    if "deepglobe" in model_path.lower() and image_path is None:
        chip_match = re.search(r'(\d+)_sat\.jpg', original_tif_path)
    else:
        chip_match = re.search(r'chip(\d+)', original_tif_path)
    chip_id = chip_match.group(1) if chip_match else "custom"
    
    print(f"[*] Processing Image #{chip_id}")
    
    image = image_tensor.unsqueeze(0)
    
    if image_path is None:
        if "deepglobe" in model_path.lower():
            mask_path = original_tif_path.replace("_sat.jpg", "_mask.png")
            from skimage import io
            mask_np = io.imread(mask_path, as_gray=True)
            mask_true = torch.from_numpy(mask_np).float().unsqueeze(0)
        else:
            mask_true = sample['mask'].float()
    else:
        # Create a dummy ground truth mask for custom images
        mask_true = torch.zeros(1, image.shape[2], image.shape[3])
    
    # Slice RGB
    # SpaceNet MS images have 8 channels. Standard images have 3.
    if image.shape[1] >= 8:
        image_rgb = image[:, [4, 2, 1], :, :]
    elif image.shape[1] >= 3:
        image_rgb = image[:, :3, :, :]
    else:
        image_rgb = image.repeat(1, 3, 1, 1) # Grayscale to RGB
        
    # Standardize size
    import torch.nn.functional as F
    if image_path is not None:
        # For custom uploaded images, gracefully resize to 1024x1024 to prevent UNet dimensionality/OOM errors
        image_rgb = F.interpolate(image_rgb, size=(1024, 1024), mode="bilinear", align_corners=False)
    elif "deepglobe" not in model_path.lower():
        # SpaceNet imagery specific upscaling (320x320 crop -> 1280x1280 upscale)
        _, _, h, w = image_rgb.shape
        if h > 1000 or w > 1000:
            image_rgb = TF.center_crop(image_rgb, [320, 320])
        image_rgb = F.interpolate(image_rgb, size=(1280, 1280), mode="bilinear", align_corners=False)
        
    # --- CRITICAL FIX: Image Normalization ---
    batch_min = image_rgb.amin(dim=(1, 2, 3), keepdim=True)
    batch_max = image_rgb.amax(dim=(1, 2, 3), keepdim=True)
    image_rgb = (image_rgb - batch_min) / (batch_max - batch_min + 1e-8)
    
    image_rgb = image_rgb.to(device)
    
    # 4. Run Inference (Predict!)
    print("[*] Running Neural Network Prediction...")
    with torch.no_grad():
        logits = model(image_rgb) 
        prediction = torch.sigmoid(logits) 
        
    # Extract tensors for plotting
    prediction = prediction.squeeze().cpu().numpy()
    mask_true = mask_true.squeeze().cpu().numpy()
    
    # Convert image for matplotlib (Requires [H, W, C] format, normalized 0-1)
    img_display = image_rgb.squeeze().permute(1, 2, 0).cpu().numpy()
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    
    prediction_thresholded = (prediction > 0.25).astype(float)
    
    # 5. Plotting (Only if requested)
    if show_plot:
        print("[*] Generating Visualization...")
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        axs[0].imshow(img_display)
        axs[0].set_title("Input (SpaceNet Mumbai)")
        axs[0].axis('off')
        
        axs[1].imshow(mask_true, cmap='gray')
        axs[1].set_title("Ground Truth (Actual Roads)")
        axs[1].axis('off')
        
        axs[2].imshow(prediction_thresholded, cmap='magma')
        axs[2].set_title("DeepGlobe Model Prediction")
        axs[2].axis('off')
        
        plt.suptitle("Cross-Dataset Generalization Test", fontsize=16)
        plt.tight_layout()
        plt.show()
        print("[+] Done! Close the plot window to exit.")
        
    # 6. Save RGB Image and Mask to disk
    os.makedirs("resilience", exist_ok=True)
    img_save_path = "resilience/spacenet_generalize_img.jpg"
    mask_save_path = "resilience/spacenet_generalize_mask.png"
    
    img_uint8 = (img_display * 255).astype('uint8')
    mask_uint8 = (prediction_thresholded * 255).astype('uint8')
    
    Image.fromarray(img_uint8).save(img_save_path)
    Image.fromarray(mask_uint8).save(mask_save_path)
    print(f"\n[*] Saved RGB Image to {img_save_path}")
    print(f"[*] Saved Prediction Mask to {mask_save_path}")
    
    return img_save_path, mask_save_path, chip_id, original_tif_path

if __name__ == "__main__":
    test_generalization(show_plot=True)
