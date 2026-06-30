import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import rasterio

# Reuse the model definition from inference
import segmentation_models_pytorch as smp
def get_model():
    return smp.UnetPlusPlus(encoder_name="resnet50", encoder_weights=None, in_channels=3, activation=None)

print("="*60)
print("  SPACENET NATIVE GEOTIFF PIPELINE TEST")
print("="*60)

spacenet_img = r"spacenet_data\SN5_roads\train\nfs\data\cosmiq\spacenet\competitions\SN5_roads\tiles_upload\train\AOI_8_Mumbai\PS-RGB\SN5_roads_train_AOI_8_Mumbai_PS-RGB_chip979.tif"
spacenet_mask = "spacenet_mask_chip979.png"
model_path = "mumbai_finetuned_model.pth"

if not os.path.exists(spacenet_img):
    print(f"[!] Could not find SpaceNet image: {spacenet_img}")
    exit()

print(f"[*] Extracting metadata from GeoTIFF using rasterio...")
with rasterio.open(spacenet_img) as src:
    print(f"    -> Bounds: {src.bounds}")
    print(f"    -> CRS: {src.crs}")

print(f"\n[*] Initializing UNet++ AI Model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# Read TIFF correctly using rasterio to ensure pure RGB bands
with rasterio.open(spacenet_img) as src:
    img_array = src.read([1, 2, 3]) # SpaceNet PS-RGB bands (R, G, B)
    # Transpose to (H, W, 3) for PIL
    img_array = np.transpose(img_array, (1, 2, 0))

# 1. MAKE IT A NORMAL RGB IMAGE FIRST
# Apply a global 2nd-98th percentile stretch to normalize brightness to match DeepGlobe
img_array = img_array.astype(np.float32)
p2, p98 = np.percentile(img_array, (2, 98))
img_array = np.clip((img_array - p2) / (p98 - p2 + 1e-5), 0, 1) * 255
img_array = img_array.astype(np.uint8)

rgb_jpg_path = "temp_rgb_input.jpg"
Image.fromarray(img_array).save(rgb_jpg_path, quality=100)
print(f"[*] Converted GeoTIFF to standard RGB JPEG: {rgb_jpg_path}")

# 2. RUN MODEL ON THE RGB JPEG (Exactly like DeepGlobe inference)
image = Image.open(rgb_jpg_path).convert("RGB")

# --- CRITICAL FIX 2: SPATIAL RESOLUTION MATCHING ---
# The model was trained on data where roads are much thinner!
# SpaceNet PS-RGB is 0.3m/px (1300x1300). DeepGlobe/MS is much lower res.
# We must downscale the image by 4x to match the spatial scale the model expects!
orig_w, orig_h = image.size
scaled_w, scaled_h = orig_w // 4, orig_h // 4
image_small = image.resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)

img_tensor = TF.to_tensor(image_small).unsqueeze(0).to(device)

# U-Net++ requires height/width to be divisible by 32. 
_, _, h, w = img_tensor.shape
pad_h = (32 - h % 32) % 32
pad_w = (32 - w % 32) % 32
import torch.nn.functional as F
img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h))

with torch.no_grad():
    raw_logits = model(img_tensor)
    probs = torch.sigmoid(raw_logits).squeeze().cpu().numpy()
    
# Crop the padding off the prediction
probs = probs[:h, :w]

# Upscale the probability mask back to the original 1300x1300 SpaceNet resolution!
probs_img = Image.fromarray((probs * 255).astype(np.uint8))
probs_img_large = probs_img.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
probs_large = np.array(probs_img_large) / 255.0

# Convert probabilties to a binary mask image
binary_mask = (probs_large > 0.25).astype(np.uint8) * 255
Image.fromarray(binary_mask).save(spacenet_mask)
print(f"[*] Saved raw prediction mask to {spacenet_mask}")

print(f"\n[*] Triggering Graph Resilience module dynamically...")
# We will temporarily edit the execution block of graph_resilience.py 
# to use our SpaceNet image, and then run it.

with open("graph_resilience.py", "r") as f:
    code = f.read()

# Swap the image paths in the execution block
code = code.replace(
    'test_img_path = f"archive/train/{img}_sat.jpg"', 
    f'test_img_path = r"{spacenet_img}"'
)
code = code.replace(
    'test_mask_path = f"archive/train/{img}_mask.png"', 
    f'test_mask_path = "{spacenet_mask}"'
)

with open("graph_resilience_spacenet.py", "w") as f:
    f.write(code)

print("\n[*] Executing Graph Resilience Analysis...")
os.system("route_env\\Scripts\\python.exe graph_resilience_spacenet.py")
