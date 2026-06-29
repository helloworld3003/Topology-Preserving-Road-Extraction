import numpy as np
import networkx as nx
import sknw
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import io
import random
import copy
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ---------------------------------------------------------
# 1. Graph Construction Layer
# ---------------------------------------------------------
def mask_to_graph(mask_array):
    """
    Converts a binary road segmentation mask into a networkx Graph.
    Uses skeletonization to thin the mask to 1-pixel wide paths.
    """
    # Ensure binary format (values > 0)
    binary_mask = (mask_array > 0).astype(np.uint8)
    
    # Extract the single-pixel wide centerline network
    skeleton = skeletonize(binary_mask)
    
    # Build the networkx graph from the skeleton
    # Nodes are intersections, edges are road segments
    graph = sknw.build_sknw(skeleton)
    
    return graph, skeleton

# ---------------------------------------------------------
# 2. Structural Chokepoint Analysis
# ---------------------------------------------------------
def calculate_centrality(graph):
    """
    Calculates Betweenness Centrality for nodes and edges to mathematically 
    identify which road segments act as critical bottlenecks.
    """
    # Calculate edge betweenness centrality
    # (weight='weight' uses the physical length of the road segment sknw provides)
    edge_centrality = nx.edge_betweenness_centrality(graph, weight='weight')
    
    # Map centrality values back onto the graph attributes
    nx.set_edge_attributes(graph, edge_centrality, 'betweenness')
    
    # Calculate node betweenness centrality (for intersections)
    node_centrality = nx.betweenness_centrality(graph, weight='weight')
    nx.set_node_attributes(graph, node_centrality, 'betweenness')
    
    return graph

# ---------------------------------------------------------
# 3. Disaster & Percolation Simulation
# ---------------------------------------------------------
def simulate_disaster(graph, deletion_fraction=0.2, mode="random"):
    """
    Simulates network failure by removing edges.
    modes: 
      - "random" (unpredictable disruption like floods)
      - "targeted" (removes highest centrality structural chokepoints first)
    """
    # Work on a copy of the graph
    g_sim = graph.copy()
    
    # Pre-disaster stats
    initial_components = nx.number_connected_components(g_sim)
    if initial_components > 0:
        # Giant Connected Component (GCC) size
        initial_gcc = max(len(c) for c in nx.connected_components(g_sim))
    else:
        initial_gcc = 0
        
    edges = list(g_sim.edges(data=True))
    num_to_remove = int(len(edges) * deletion_fraction)
    
    # Select edges to remove based on mode
    if mode == "random":
        edges_to_remove = random.sample(edges, num_to_remove)
    elif mode == "targeted":
        # Sort edges by betweenness centrality (highest first)
        edges_sorted = sorted(edges, key=lambda x: x[2].get('betweenness', 0), reverse=True)
        edges_to_remove = edges_sorted[:num_to_remove]
    else:
        raise ValueError("mode must be 'random' or 'targeted'")
        
    # Execute attack
    for u, v, data in edges_to_remove:
        if g_sim.has_edge(u, v):
            g_sim.remove_edge(u, v)
            
    # Post-disaster stats
    post_components = nx.number_connected_components(g_sim)
    if post_components > 0:
        post_gcc = max(len(c) for c in nx.connected_components(g_sim))
    else:
        post_gcc = 0
        
    # Calculate fragmentation severity
    gcc_drop_percentage = ((initial_gcc - post_gcc) / initial_gcc) * 100 if initial_gcc > 0 else 0
    
    metrics = {
        "mode": mode,
        "deletion_fraction": deletion_fraction,
        "initial_components": initial_components,
        "post_components": post_components,
        "initial_gcc_size": initial_gcc,
        "post_gcc_size": post_gcc,
        "gcc_size_drop_percent": gcc_drop_percentage
    }
    
    return g_sim, metrics

# ---------------------------------------------------------
# 4. Geospatial Visualization Layer
# ---------------------------------------------------------
def visualize_network_resilience(original_img, mask_img, graph, metrics=None, save_path=None):
    """
    Overlays the network graph on top of the original image,
    highlighting critical chokepoints dynamically based on betweenness centrality.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot 1: Original Image
    axes[0].imshow(original_img)
    axes[0].set_title("Original Satellite Imagery", fontsize=16)
    axes[0].axis('off')
    
    # Plot 2: Binary Mask
    axes[1].imshow(mask_img, cmap='gray')
    axes[1].set_title("Predicted Road Topology", fontsize=16)
    axes[1].axis('off')
    
    # Plot 3: Network Resilience Graph Overlay
    axes[2].imshow(original_img)
    axes[2].set_title("Structural Chokepoint Analysis", fontsize=16)
    axes[2].axis('off')
    
    # Extract edge centralities for dynamic color mapping (Heatmap style)
    edge_centralities = [data.get('betweenness', 0) for u, v, data in graph.edges(data=True)]
    
    if edge_centralities:
        max_cent = max(edge_centralities)
        min_cent = min(edge_centralities)
        if max_cent == min_cent:
            max_cent = min_cent + 1e-6
    else:
        max_cent = 1
        min_cent = 0

    norm = mcolors.Normalize(vmin=min_cent, vmax=max_cent)
    cmap = cm.inferno # Bright yellow/white = Critical Bottleneck
    
    # Draw graph edges onto the image
    for (u, v, data) in graph.edges(data=True):
        pts = data['pts'] # The spatial pixel coordinates generated by sknw
        cent = data.get('betweenness', 0)
        color = cmap(norm(cent))
        
        # Thicken lines dynamically for higher centrality
        lw = 1.5 + 4.0 * ((cent - min_cent) / (max_cent - min_cent))
        axes[2].plot(pts[:, 1], pts[:, 0], color=color, linewidth=lw, alpha=0.85)
        
    # Draw graph nodes (intersections)
    node_pts = np.array([graph.nodes[node]['o'] for node in graph.nodes()])
    if len(node_pts) > 0:
        axes[2].scatter(node_pts[:, 1], node_pts[:, 0], c='cyan', s=15, zorder=5)
        
    # Add Legend for the intersection points
    axes[2].scatter([], [], c='cyan', s=30, label='Intersections (Nodes)')
    axes[2].legend(loc='lower right', facecolor='black', labelcolor='white', framealpha=0.8)
    
    # Add Colorbar for the edge heatmap
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Structural Vulnerability (Centrality)', fontsize=12, color='black')
        
    # Add Disaster Metrics Overlay
    if metrics:
        info_text = (
            f"DISASTER SIMULATION METRICS\n"
            f"----------------------------------------\n"
            f"Attack Type: {metrics['deletion_fraction']*100:.0f}% {metrics['mode'].capitalize()} Failure\n"
            f"Pre-Attack Subgraphs: {metrics['initial_components']}\n"
            f"Post-Attack Subgraphs: {metrics['post_components']} (Fragmentation)\n"
            f"GCC Connectivity Drop: {metrics['gcc_size_drop_percent']:.1f}%"
        )
        axes[2].text(0.02, 0.98, info_text, transform=axes[2].transAxes, 
                     fontsize=14, color='white', verticalalignment='top',
                     bbox=dict(facecolor='black', alpha=0.75, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[*] Visualization saved to: {save_path}")

# ---------------------------------------------------------
# Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print("  NETWORK RESILIENCE & DISASTER SIMULATION (GRAPH THEORY)")
    print("="*70)
    
    # NOTE: To test this locally, place a valid satellite image and its corresponding mask here.
    # For now, this is a template block ready for your specific image paths.
    img=5668
    test_img_path = "archive/train/"+str(img)+"_sat.jpg"   # Adjust to an actual image you want to test
    test_mask_path = "archive/train/"+str(img)+"_mask.png" # Adjust to the prediction or ground truth mask
    
    if os.path.exists(test_img_path) and os.path.exists(test_mask_path):
        print(f"[*] Loading satellite image: {test_img_path}")
        original_img = io.imread(test_img_path)
        
        print(f"[*] Loading segmentation mask: {test_mask_path}")
        mask_img = io.imread(test_mask_path, as_gray=True)
        
        print("[*] 1. Constructing Mathematical Graph from Mask (sknw)...")
        graph, skeleton = mask_to_graph(mask_img)
        print(f"    -> Identified {graph.number_of_nodes()} intersections (Nodes)")
        print(f"    -> Identified {graph.number_of_edges()} road segments (Edges)")
        
        if graph.number_of_nodes() > 0:
            print("[*] 2. Calculating Structural Chokepoints (Betweenness Centrality)...")
            graph = calculate_centrality(graph)
            
            print("\n[*] 3. Running Disaster & Percolation Simulations...")
            
            # Simulate 15% Random Failure (e.g., flash flooding)
            print("  --- SIMULATION: 15% RANDOM FAILURE ---")
            rand_graph, rand_metrics = simulate_disaster(graph, deletion_fraction=0.15, mode="random")
            print(f"    Initial Subgraphs (Islands) : {rand_metrics['initial_components']}")
            print(f"    Post-Disaster Subgraphs     : {rand_metrics['post_components']}")
            print(f"    Network Fragmentation       : {rand_metrics['gcc_size_drop_percent']:.1f}% capacity lost")
            
            # Simulate 15% Targeted Attack (e.g., strategic structural collapse)
            print("\n  --- SIMULATION: 15% TARGETED ATTACK ---")
            targ_graph, targ_metrics = simulate_disaster(graph, deletion_fraction=0.15, mode="targeted")
            print(f"    Initial Subgraphs (Islands) : {targ_metrics['initial_components']}")
            print(f"    Post-Disaster Subgraphs     : {targ_metrics['post_components']}")
            print(f"    Network Fragmentation       : {targ_metrics['gcc_size_drop_percent']:.1f}% capacity lost")
            
            print("\n[*] 4. Generating Geospatial Visualization Layer...")
            os.makedirs("resilience", exist_ok=True)
            save_file = f"resilience/Network_Resilience_Analysis_{img}.png"
            visualize_network_resilience(original_img, mask_img, graph, metrics=targ_metrics, save_path=save_file)
            
            print("\n[+] Graph Resilience Analysis Complete!")
        else:
            print("[!] Warning: No valid road network found in the mask to build a graph.")
    else:
        print(f"[!] Example files not found.")
        print(f"    Looked for: {test_img_path}")
        print("    Please point test_img_path and test_mask_path to real images inside the script.")
