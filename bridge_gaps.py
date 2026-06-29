import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from PIL import Image
from skimage.morphology import skeletonize, dilation, disk
from scipy.signal import convolve2d
from scipy.spatial import cKDTree
from skimage.draw import line

# ---------------------------------------------------------
# 1. Model Initialization
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
# 2. Endpoint Bridging Algorithm
# ---------------------------------------------------------
def find_endpoints(skeleton):
    """
    Finds endpoints of a 1-pixel wide skeleton.
    Uses a convolution kernel to check the 8-neighborhood.
    A skeleton pixel (10) with exactly 1 neighbor (1) will sum to 11.
    """
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    
    neighbor_count = convolve2d(skeleton.astype(np.uint8), kernel, mode='same')
    endpoints = np.argwhere(neighbor_count == 11)
    return endpoints

def bridge_skeleton_gaps(binary_mask, max_gap=25):
    """
    Finds "dead ends" in the road mask and draws a mathematical line to the 
    nearest road skeleton pixel if it's within the `max_gap` distance.
    """
    print("[*] Bridging structural gaps...")
    # 1. Extract the 1-pixel wide skeleton from the AI prediction
    skeleton = skeletonize(binary_mask)
    
    # 2. Find the dead ends
    endpoints = find_endpoints(skeleton)
    print(f"    -> Found {len(endpoints)} disconnected dead ends in the road network.")
    
    bridged_mask = binary_mask.copy().astype(np.uint8)
    
    if len(endpoints) == 0:
        return bridged_mask
        
    skel_coords = np.argwhere(skeleton > 0)
    tree = cKDTree(skel_coords)
    
    bridges_drawn = 0
    # 3. For each endpoint, find the nearest connection point
    for ep in endpoints:
        # Search in a radius up to max_gap
        # k=100 ensures we find neighbors that aren't literally the endpoint's own road segment
        dists, idxs = tree.query(ep, k=min(100, len(skel_coords)), distance_upper_bound=max_gap)
        
        valid_connection = None
        for d, i in zip(dists, idxs):
            if d == np.inf:
                continue
            # Must be at least a few pixels away to avoid bridging to its own adjacent pixels
            if d > 5:
                valid_connection = skel_coords[i]
                break
                
        # 4. Draw the mathematical bridge between them
        if valid_connection is not None:
            rr, cc = line(ep[0], ep[1], valid_connection[0], valid_connection[1])
            bridged_mask[rr, cc] = 1
            bridges_drawn += 1
            
    print(f"    -> Successfully drew {bridges_drawn} new bridges!")
    
    # 5. Thicken the new 1-pixel bridges slightly to match the actual road thickness
    bridges_only = bridged_mask ^ binary_mask
    thick_bridges = dilation(bridges_only, disk(2))
    
    final_mask = np.logical_or(binary_mask, thick_bridges).astype(np.uint8)
    return final_mask

# ---------------------------------------------------------
# 3. Execution Engine
# ---------------------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print("  ROAD GAP-FILLING POST-PROCESSOR (ENDPOINT HEURISTIC)")
    print("="*70)
    
    # We will test this on a known image from your archive.
    # To test the specific image from 'rank 17', look up its ID in the training logs or cherry-picker, 
    # and swap the img_id here!
    
    img_id = "4227" 
    img_path = f"archive/train/{img_id}_sat.jpg"
    model_path = "deepglobe_finetuned_model.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Device: {device}")
    
    if not os.path.exists(img_path):
        print(f"[!] Could not find {img_path}")
        exit()
        
    print(f"[*] Loading model from {model_path}...")
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Run Inference
    print(f"[*] Running raw inference on image {img_id}...")
    image = Image.open(img_path).convert("RGB")
    img_tensor = TF.to_tensor(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        raw_logits = model(img_tensor)
        probs = torch.sigmoid(raw_logits).squeeze().cpu().numpy()
        
    raw_mask = (probs > 0.25).astype(np.uint8)
    
    # Run Post-Processing
    bridged_mask = bridge_skeleton_gaps(raw_mask, max_gap=25)
    
    # Visualization
    print("[*] Generating side-by-side comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Satellite Imagery", fontsize=16)
    axes[0].axis('off')
    
    axes[1].imshow(raw_mask, cmap='gray')
    axes[1].set_title("Raw AI Prediction (Notice tiny gaps)", fontsize=16)
    axes[1].axis('off')
    
    # To make the bridges highly visible for the presentation, we can color the bridged mask!
    # Let's plot the base mask in gray, and highlight the new bridges in bright red.
    bridges_only = bridged_mask ^ raw_mask
    color_mask = np.zeros((raw_mask.shape[0], raw_mask.shape[1], 3), dtype=np.uint8)
    color_mask[raw_mask == 1] = [255, 255, 255] # Base roads are white
    color_mask[bridges_only == 1] = [255, 0, 0] # New bridges are red!
    
    axes[2].imshow(color_mask)
    axes[2].set_title("Post-Processed (Bridges Highlighted in Red)", fontsize=16)
    axes[2].axis('off')
    
    os.makedirs("Documentation/resilience", exist_ok=True)
    save_path = "Documentation/resilience/Bridged_Gaps_Comparison.png"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"[*] Visual comparison saved to: {save_path}")
    print("[+] Done!")
