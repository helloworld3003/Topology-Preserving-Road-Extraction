import os
from skimage import io

# Import the generalization inference pipeline
from inference_generalize import test_generalization
# Import the resilience analysis functions
from graph_resilience import (
    mask_to_graph,
    calculate_centrality,
    simulate_cascading_failure,
    project_graph_to_geo,
    export_to_geojson,
    visualize_network_resilience
)

if __name__ == "__main__":
    print("="*70)
    print("  NETWORK RESILIENCE & DISASTER SIMULATION (GRAPH THEORY)")
    print("="*70)
    
    print("[*] Automatically generating a new Multi-Spectral SpaceNet Mask...")
    img_save_path, mask_save_path, chip_id = test_generalization()
    
    if os.path.exists(img_save_path) and os.path.exists(mask_save_path):
        print(f"\n[*] Loading satellite image crop: {img_save_path}")
        original_img = io.imread(img_save_path)
        img_h, img_w = original_img.shape[:2]
        
        print(f"[*] Loading segmentation mask: {mask_save_path}")
        mask_img = io.imread(mask_save_path, as_gray=True)
        
        print("[*] 1. Constructing Mathematical Graph from Mask (sknw)...")
        graph, skeleton = mask_to_graph(mask_img)
        print(f"    -> Identified {graph.number_of_nodes()} intersections (Nodes)")
        print(f"    -> Identified {graph.number_of_edges()} road segments (Edges)")
        
        if graph.number_of_nodes() > 0:
            print("[*] 2. Calculating Structural Chokepoints (Betweenness Centrality)...")
            graph = calculate_centrality(graph)
            
            print("\n[*] 3. Running Dynamic Cascading Overload Simulation...")
            export_graph, active_graph, cascade_metrics = simulate_cascading_failure(graph, tolerance=1.5)
            
            print(f"    - Cascade Depth: {cascade_metrics['cascade_steps']} chain-reaction steps")
            print(f"    - Total Roads Failed: {cascade_metrics['failed_edges_total']}")
            print(f"    - Network Fragmentation: {cascade_metrics['gcc_size_drop_percent']:.1f}% capacity lost")
            
            print("\n[*] 4. Projecting Graph to Earth Coordinates...")
            export_graph = project_graph_to_geo(export_graph, img_width=img_w, img_height=img_h, img_path=img_save_path)
            
            os.makedirs("resilience", exist_ok=True)
            geojson_path = f"resilience/Network_Resilience_3D_MS_chip{chip_id}.geojson"
            print(f"\n[*] 5. Exporting GeoJSON for Kepler.gl: {geojson_path}")
            export_to_geojson(export_graph, geojson_path)
            
            print(f"[*] 6. Generating 2D Geospatial Visualization Layer...")
            save_file = f"resilience/Network_Resilience_Analysis_MS_chip{chip_id}.png"
            visualize_network_resilience(original_img, mask_img, export_graph, metrics=cascade_metrics, save_path=save_file)
            
            print("\n[+] Graph Resilience Analysis & Geo Export Complete!")
        else:
            print("[!] Warning: No valid road network found in the mask to build a graph.")
            
        print("\n[*] Cleaning up temporary prediction files...")
        if os.path.exists(img_save_path): os.remove(img_save_path)
        if os.path.exists(mask_save_path): os.remove(mask_save_path)
        print("[*] Cleanup complete.")
    else:
        print(f"[!] Mask generation failed.")
