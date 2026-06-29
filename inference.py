import os
import random
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
import glob
from PIL import Image

# ---------------------------------------------------------
# 1. Initialize Identical Model Architecture
# ---------------------------------------------------------
def get_model():
    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights=None,       # No need for ImageNet weights, we are loading our own!
        in_channels=3,              # 3-band RGB (from our train.py fix)
        # Important: Remove 'sigmoid' activation here because train.py was updated to output raw logits!
        activation=None
    )
    return model

# ---------------------------------------------------------
# 2. Inference & Visualization Logic
# ---------------------------------------------------------
def visualize_prediction(model_path="mumbai_finetuned_model.pth"):
    print("\n========== INFERENCE ENGINE ==========") 
    print(f"[*] Initializing Pipeline...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Device: {device}")
    
    # 1. Load Model
    model = get_model()
    if not os.path.exists(model_path):
        print(f"\n[!] ERROR: Cannot find {model_path}!")
        print("    Did you run train.py first to generate the weights?")
        return
        
    print(f"[*] Loading Brain Data from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    
    # CRITICAL: Put model in evaluation mode (turns off dropout, locks batch normalization)
    model.eval() 
    
    # 2. Load a Sample from the Dataset
    print("[*] Fetching a test image from DeepGlobe (archive/train)...")
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive", "train")
    all_images = sorted(glob.glob(os.path.join(data_dir, "*_sat.jpg")))
    
    if not all_images:
        print(f"[!] ERROR: No DeepGlobe *_sat.jpg images found in {data_dir}")
        return
        
    # Pick a random DeepGlobe image
    random_idx = random.randint(0, len(all_images) - 1)
    img_path = all_images[random_idx]
    mask_path = img_path.replace("_sat.jpg", "_mask.png")
    
    print(f"[*] Fetched Image: {os.path.basename(img_path)}")
    
    # Load with PIL exactly like train2.py
    image_pil = Image.open(img_path).convert("RGB")
    mask_pil = Image.open(mask_path).convert("L") if os.path.exists(mask_path) else None
    
    image = TF.to_tensor(image_pil).unsqueeze(0) # [1, 3, H, W]
    
    # DeepGlobe images are native RGB, so NO channel slicing needed like SpaceNet!
    image_rgb = image 
    
    # Center Crop to 320x320
    image_rgb = TF.center_crop(image_rgb, [320, 320])
    
    if mask_pil is not None:
        mask_true = TF.to_tensor(mask_pil)
        mask_true = TF.center_crop(mask_true, [320, 320])
    else:
        mask_true = torch.zeros((1, 320, 320))
    
    # --- CRITICAL FIX: Image Normalization ---
    batch_min = image_rgb.amin(dim=(1, 2, 3), keepdim=True)
    batch_max = image_rgb.amax(dim=(1, 2, 3), keepdim=True)
    image_rgb = (image_rgb - batch_min) / (batch_max - batch_min + 1e-8)
    
    image_rgb = image_rgb.to(device)
    
    # 4. Run Inference (Predict!)
    print("[*] Running Neural Network Prediction...")
    with torch.no_grad(): # Don't calculate gradients during testing (saves memory & time)
        logits = model(image_rgb) # Shape: [1, 1, 320, 320]
        prediction = torch.sigmoid(logits) # Convert raw logits back to probabilities!
        
    # Extract tensors for plotting
    prediction = prediction.squeeze().cpu().numpy()
    mask_true = mask_true.squeeze().cpu().numpy()
    
    # Convert image for matplotlib (Requires [H, W, C] format, normalized 0-1)
    img_display = image_rgb.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Normalize RGB image colors to [0, 1] for visual display
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    
    # 5. Apply a threshold 
    # The model outputs a probability (0.0 to 1.0). If it's over 50% sure it's a road, we keep it!
    prediction_thresholded = (prediction > 0.5).astype(float)
    
    # 6. Plotting
    print("[*] Generating Visualization...")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(img_display)
    axs[0].set_title("Input (RGB Satellite Image)")
    axs[0].axis('off')
    
    axs[1].imshow(mask_true, cmap='gray')
    axs[1].set_title("Ground Truth (Actual Roads)")
    axs[1].axis('off')
    
    # We use the 'magma' color map so the predicted roads glow bright yellow/purple!
    axs[2].imshow(prediction_thresholded, cmap='magma')
    axs[2].set_title("Model Prediction (10 Epochs)")
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("[+] Done! Close the plot window to exit.")

if __name__ == "__main__":
    visualize_prediction()
