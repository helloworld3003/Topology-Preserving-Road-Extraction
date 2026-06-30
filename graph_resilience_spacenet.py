import os
from skimage import io

# Import the generalization inference pipeline
from inference_generalize import test_generalization
# Import the resilience analysis functions
from graph_resilience import (
    mask_to_graph,
    calculate_centrality,
    simulate_cascading_failure,
    simulate_random_attack,
    project_graph_to_geo,
    export_to_geojson,
    visualize_network_resilience
)
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mumbai_finetuned_model.pth')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--attack', type=str, default='targeted', choices=['targeted', 'random'])
    parser.add_argument('--remove_pct', type=float, default=0.15)
    args = parser.parse_args()

    print("="*70)
    print("  NETWORK RESILIENCE & DISASTER SIMULATION (GRAPH THEORY)")
    print("="*70)
    
    print("[*] Automatically generating a new Multi-Spectral SpaceNet Mask...")
    img_save_path, mask_save_path, chip_id, original_tif_path = test_generalization(model_path=args.model, image_path=args.image)
    
    if os.path.exists(img_save_path) and os.path.exists(mask_save_path):
        print(f"\n[*] Loading satellite image crop: {img_save_path}")
        original_img = io.imread(img_save_path)
        # The inference output is now upscaled to 1280x1280 for spatial matching
        img_w, img_h = 1280, 1280
        
        print(f"[*] Loading segmentation mask: {mask_save_path}")
        mask_img = io.imread(mask_save_path, as_gray=True)
        
        print("[*] 1. Constructing Mathematical Graph from Mask (sknw)...")
        graph, skeleton = mask_to_graph(mask_img)
        print(f"    -> Identified {graph.number_of_nodes()} intersections (Nodes)")
        print(f"    -> Identified {graph.number_of_edges()} road segments (Edges)")
        
        if graph.number_of_nodes() > 0:
            print("[*] 2. Calculating Structural Chokepoints (Betweenness Centrality)...")
            graph = calculate_centrality(graph)
            
            if args.attack == 'targeted':
                print("\n[*] 3. Running Targeted Attack (Cascading Overload)...")
                export_graph, active_graph, metrics = simulate_cascading_failure(graph, tolerance=1.5)
                print(f"    - Cascade Depth: {metrics.get('cascade_steps', 0)} chain-reaction steps")
                print(f"    - Total Roads Failed: {metrics.get('failed_edges_total', 0)}")
                print(f"    - Network Fragmentation: {metrics.get('capacity_lost_pct', 0):.1f}% capacity lost")
            else:
                print(f"\n[*] 3. Running Random Attack (Removing {args.remove_pct*100}% of nodes)...")
                active_graph, metrics = simulate_random_attack(graph, remove_pct=args.remove_pct)
                export_graph = graph.copy() # Base graph for export
                print(f"    - Total Nodes Failed: {metrics.get('failed_nodes', 0)}")
                print(f"    - Islands Created: {metrics.get('islands_created', 0)}")
                print(f"    - Network Fragmentation: {metrics.get('capacity_lost_pct', 0):.1f}% capacity lost")
            
            print("\n[*] 4. Projecting Graph to Earth Coordinates...")
            export_graph = project_graph_to_geo(export_graph, img_width=img_w, img_height=img_h, img_path=original_tif_path)
            
            os.makedirs("resilience", exist_ok=True)
            geojson_path = f"resilience/Network_Resilience_3D_MS_chip{chip_id}.geojson"
            print(f"\n[*] 5. Exporting GeoJSON for Kepler.gl: {geojson_path}")
            export_to_geojson(export_graph, geojson_path)
            
            print(f"[*] 6. Generating 2D Geospatial Visualization Layer...")
            save_file = f"resilience/Network_Resilience_Analysis_MS_chip{chip_id}.png"
            visualize_network_resilience(original_img, mask_img, export_graph, metrics=metrics, save_path=save_file)
            
            print("\n[+] Graph Resilience Analysis & Geo Export Complete!")
        else:
            print("[!] Warning: No valid road network found in the mask to build a graph.")
    else:
        print("[!] Error: Could not generate predictions. Check inference logs.")
