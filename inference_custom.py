import os
import sys
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
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
# 2. Custom Image Inference Logic
# ---------------------------------------------------------
def test_custom_image(image_path, model_path="deepglobe_road_model.pth"):
    print("\n========== CUSTOM SATELLITE IMAGE TEST ==========")
    print(f"[*] Testing DeepGlobe-trained model on your Google Maps screenshot!")
    
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
    
    # 2. Load the User's Custom Image
    if not os.path.exists(image_path):
        print(f"\n[!] ERROR: Cannot find custom image at {image_path}!")
        return
        
    print(f"[*] Loading Image: {os.path.basename(image_path)}")
    image_pil = Image.open(image_path).convert("RGB")
    
    # 3. Preprocess
    image = TF.to_tensor(image_pil).unsqueeze(0) # [1, 3, H, W]
    
    # --- CRITICAL FIX: SCALE VARIANCE ---
    # Neural Networks are extremely sensitive to zoom levels. 
    # If we 'resize' the image, a 15-pixel wide highway becomes a 150-pixel wide gray blob,
    # and the CNN's mathematical filters can no longer recognize it as a road.
    # Instead of resizing, we will pad/crop it to the nearest multiple of 32 to keep the native scale!
    
    _, _, h, w = image.shape
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32
    
    image_rgb = TF.center_crop(image, [new_h, new_w])
    
    # --- Image Normalization ---
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
    
    # Convert image for matplotlib (Requires [H, W, C] format, normalized 0-1)
    img_display = image_rgb.squeeze().permute(1, 2, 0).cpu().numpy()
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    
    # Instead of aggressively destroying data with a > 0.5 threshold, 
    # let's look at the RAW Probability Heatmap to see if the AI suspects anything!
    prediction_heatmap = prediction 
    
    # 5. Plotting (Only 2 columns this time, no Ground Truth mask!)
    print("[*] Generating Visualization...")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    axs[0].imshow(img_display)
    axs[0].set_title("Input (Your Custom Google Maps Image)")
    axs[0].axis('off')
    
    # Plot the raw probabilities using the 'magma' colormap
    im = axs[1].imshow(prediction_heatmap, cmap='magma', vmin=0, vmax=1)
    axs[1].set_title("Raw AI Confidence Heatmap")
    axs[1].axis('off')
    plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    
    plt.suptitle("Custom Wild Image Inference", fontsize=16)
    plt.tight_layout()
    plt.show()
    print("[+] Done! Close the plot window to exit.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_custom.py <path_to_image>")
    else:
        test_custom_image(sys.argv[1])
