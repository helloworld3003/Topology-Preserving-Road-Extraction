import os
import random
import torch
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
# 2. Generalization Inference Logic
# ---------------------------------------------------------
def test_generalization(model_path="mumbai_finetuned_model.pth"):
    print("\n========== CROSS-DATASET GENERALIZATION TEST ==========")
    print(f"[*] Testing how well the DeepGlobe-trained model performs on SpaceNet 5 data!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Device: {device}")
    
    # 1. Load Model (Trained on DeepGlobe)
    model = get_model()
    if not os.path.exists(model_path):
        print(f"\n[!] ERROR: Cannot find {model_path}!")
        return
        
    print(f"[*] Loading Brain Data from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() 
    
    # 2. Load a Sample from the SpaceNet 5 (Mumbai) Dataset
    print("[*] Fetching an unseen test image from SpaceNet 5 (Mumbai)...")
    dataset = SpaceNet5(root="spacenet_data", aois=[8], split="train", download=False)
    
    random_idx = random.randint(0, len(dataset) - 1)
    sample = dataset[random_idx] 
    print(f"[*] Fetched SpaceNet Image #{random_idx}")
    
    # 3. Preprocess SpaceNet exactly like we did during the Mumbai phase
    image = sample['image'].float().unsqueeze(0) 
    mask_true = sample['mask'].float()           
    
    # Slice RGB (Channels 4, 2, 1) for SpaceNet multi-spectral images
    image_rgb = image[:, [4, 2, 1], :, :]
    
    # Center Crop to 320x320
    image_rgb = TF.center_crop(image_rgb, [320, 320])
    mask_true = TF.center_crop(mask_true, [320, 320])
    
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
    
    prediction_thresholded = (prediction > 0.5).astype(float)
    
    # 5. Plotting
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

if __name__ == "__main__":
    test_generalization()
