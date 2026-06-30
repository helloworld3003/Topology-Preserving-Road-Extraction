import numpy as np
import networkx as nx
import sknw
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import io
import random
import copy
import os
import json
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
    binary_mask = (mask_array > 0).astype(np.uint8)
    skeleton = skeletonize(binary_mask)
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
    edge_centrality = nx.edge_betweenness_centrality(graph, weight='weight')
    nx.set_edge_attributes(graph, edge_centrality, 'betweenness')
    
    node_centrality = nx.betweenness_centrality(graph, weight='weight')
    nx.set_node_attributes(graph, node_centrality, 'betweenness')
    
    return graph

# ---------------------------------------------------------
# 3. Geo-Spatial Projection Layer
# ---------------------------------------------------------
def project_graph_to_geo(G, img_width=1024, img_height=1024, img_path=None):
    """
    Translates pixel-based graph nodes to real-world coordinates.
    If a GeoTIFF image is provided, it extracts the exact bounds using rasterio.
    Otherwise, it uses an affine bounding-box transformation over Mumbai, India.
    """
    # Fallback to a realistic 500m x 500m bounding box in Central Mumbai
    LON_MIN, LON_MAX = 72.9400, 72.9450 
    LAT_MIN, LAT_MAX = 19.1050, 19.1100 
    
    if img_path and str(img_path).lower().endswith('.tif'):
        try:
            import rasterio
            with rasterio.open(img_path) as src:
                bounds = src.bounds
                LON_MIN, LON_MAX = bounds.left, bounds.right
                LAT_MIN, LAT_MAX = bounds.bottom, bounds.top
                img_width = src.width
                img_height = src.height
                print(f"[*] Successfully extracted real GPS metadata from GeoTIFF: {img_path}")
                print(f"    -> Bounds: Lon({LON_MIN:.4f}, {LON_MAX:.4f}), Lat({LAT_MIN:.4f}, {LAT_MAX:.4f})")
        except Exception as e:
            print(f"[!] Failed to read rasterio metadata from {img_path}: {e}")
            print(f"[*] Falling back to default Mumbai bounding box...")
    else:
        print("[*] Projecting pixel graph onto Earth coordinates using default Mumbai bounding box...")
    
    for node, data in G.nodes(data=True):
        y_pixel, x_pixel = data['o']
        lon = LON_MIN + (x_pixel / img_width) * (LON_MAX - LON_MIN)
        lat = LAT_MAX - (y_pixel / img_height) * (LAT_MAX - LAT_MIN)
        G.nodes[node]['lon'] = lon
        G.nodes[node]['lat'] = lat
        
    for u, v, data in G.edges(data=True):
        geo_pts = []
        for pt in data['pts']:
            y_pixel, x_pixel = pt
            lon = LON_MIN + (x_pixel / img_width) * (LON_MAX - LON_MIN)
            lat = LAT_MAX - (y_pixel / img_height) * (LAT_MAX - LAT_MIN)
            geo_pts.append([lon, lat])
        data['geo_pts'] = geo_pts
        
    return G

# ---------------------------------------------------------
# 4. Disaster Simulation (Cascading Overload)
# ---------------------------------------------------------
def simulate_cascading_failure(graph, tolerance=1.5):
    """
    Simulates dynamic load redistribution. When a critical road fails, 
    its traffic dumps onto neighboring roads, potentially causing them 
    to exceed their capacity and fail in a chain reaction.
    """
    export_graph = graph.copy()
    
    for u, v, data in export_graph.edges(data=True):
        load = data.get('betweenness', 0.0)
        data['initial_load'] = load
        data['capacity'] = load * tolerance
        data['status'] = 'active'
        
    active_graph = export_graph.copy()
    
    initial_components = nx.number_connected_components(active_graph)
    initial_gcc = max(len(c) for c in nx.connected_components(active_graph)) if initial_components > 0 else 0
    
    edges = list(active_graph.edges(data=True))
    if not edges:
        return export_graph, active_graph, {}
        
    # Trigger Failure
    highest_edge = max(edges, key=lambda x: x[2].get('betweenness', 0))
    active_graph.remove_edge(highest_edge[0], highest_edge[1])
    export_graph[highest_edge[0]][highest_edge[1]]['status'] = 'trigger_destroyed'
    failed_edges_count = 1
    
    # Cascade Loop
    cascade_step = 1
    while True:
        new_centrality = nx.edge_betweenness_centrality(active_graph, weight='weight')
        nx.set_edge_attributes(active_graph, new_centrality, 'betweenness')
        
        overloaded = []
        for u, v, data in active_graph.edges(data=True):
            if data['betweenness'] > export_graph[u][v]['capacity']:
                overloaded.append((u, v))
                
        if not overloaded:
            break
            
        for u, v in overloaded:
            if active_graph.has_edge(u, v):
                active_graph.remove_edge(u, v)
                export_graph[u][v]['status'] = f'failed_step_{cascade_step}'
                failed_edges_count += 1
                
        cascade_step += 1
                
    post_components = nx.number_connected_components(active_graph)
    post_gcc = max(len(c) for c in nx.connected_components(active_graph)) if post_components > 0 else 0
    gcc_drop_percentage = ((initial_gcc - post_gcc) / initial_gcc) * 100 if initial_gcc > 0 else 0
    
    metrics = {
        "mode": "cascading_overload",
        "tolerance": tolerance,
        "failed_edges_total": failed_edges_count,
        "cascade_steps": cascade_step - 1,
        "initial_components": initial_components,
        "post_components": post_components,
        "initial_gcc": initial_gcc,
        "post_gcc": post_gcc,
        "capacity_lost_pct": gcc_drop_percentage
    }
    
    return export_graph, active_graph, metrics

def simulate_random_attack(graph, remove_pct=0.15):
    import random
    active_graph = graph.copy()
    
    initial_components = nx.number_connected_components(active_graph)
    initial_gcc = max(len(c) for c in nx.connected_components(active_graph)) if initial_components > 0 else 0
    
    nodes = list(active_graph.nodes())
    if not nodes:
        return active_graph, {}
        
    num_to_remove = max(1, int(len(nodes) * remove_pct))
    nodes_to_remove = random.sample(nodes, num_to_remove)
    
    active_graph.remove_nodes_from(nodes_to_remove)
    
    final_components = nx.number_connected_components(active_graph)
    final_gcc = max(len(c) for c in nx.connected_components(active_graph)) if final_components > 0 else 0
    
    cap_lost = 100.0 * (initial_gcc - final_gcc) / initial_gcc if initial_gcc > 0 else 100.0
    
    return active_graph, {
        'failed_nodes': num_to_remove,
        'capacity_lost_pct': cap_lost,
        'islands_created': final_components - initial_components
    }

# ---------------------------------------------------------
# 5. GeoJSON Export Layer (Kepler.gl)
# ---------------------------------------------------------
def export_to_geojson(graph, output_path):
    """
    Exports the projected graph to a standard GeoJSON FeatureCollection.
    """
    features = []
    
    for u, v, data in graph.edges(data=True):
        if 'geo_pts' not in data:
            continue
        properties = {
            "betweenness": float(data.get('betweenness', 0.0)),
            "capacity": float(data.get('capacity', 0.0)),
            "initial_load": float(data.get('initial_load', 0.0)),
            "status": str(data.get('status', 'active'))
        }
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": data['geo_pts']
            },
            "properties": properties
        }
        features.append(feature)
        
    for node, data in graph.nodes(data=True):
        if 'lon' in data and 'lat' in data:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [data['lon'], data['lat']]
                },
                "properties": {
                    "betweenness": float(data.get('betweenness', 0.0)),
                    "node_id": int(node)
                }
            }
            features.append(feature)
            
    geojson = {"type": "FeatureCollection", "features": features}
    with open(output_path, 'w') as f:
        json.dump(geojson, f)
    print(f"[*] GeoJSON successfully exported to: {output_path}")

# ---------------------------------------------------------
# 6. Geospatial Visualization Layer (2D Plotting)
# ---------------------------------------------------------
def visualize_network_resilience(original_img, mask_img, graph, metrics=None, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    ax.imshow(original_img)
    ax.axis('off')
    
    edge_centralities = [data.get('betweenness', 0) for u, v, data in graph.edges(data=True) if data.get('status', 'active') == 'active']
    if edge_centralities:
        max_cent, min_cent = max(edge_centralities), min(edge_centralities)
        if max_cent == min_cent: max_cent = min_cent + 1e-6
    else:
        max_cent, min_cent = 1, 0

    norm = mcolors.Normalize(vmin=min_cent, vmax=max_cent)
    cmap = cm.inferno 
    
    for (u, v, data) in graph.edges(data=True):
        pts = data['pts'] 
        status = data.get('status', 'active')
        
        if status == 'active':
            cent = data.get('betweenness', 0)
            color = cmap(norm(cent))
            lw = 1.5 + 4.0 * ((cent - min_cent) / (max_cent - min_cent))
            ax.plot(pts[:, 1], pts[:, 0], color=color, linewidth=lw, alpha=0.85)
        else:
            # Destroyed Roads (Trigger or Cascade)
            ax.plot(pts[:, 1], pts[:, 0], color='red', linewidth=4.0, alpha=0.9, linestyle='--')
        
    node_pts = np.array([graph.nodes[node]['o'] for node in graph.nodes()])
    if len(node_pts) > 0:
        ax.scatter(node_pts[:, 1], node_pts[:, 0], c='cyan', s=15, zorder=5)
        
    ax.scatter([], [], c='cyan', s=30, label='Intersections (Nodes)')
    ax.legend(loc='lower right', facecolor='black', labelcolor='white', framealpha=0.8)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Structural Vulnerability (Centrality)', fontsize=12, color='black')
        
    if metrics:
        if metrics.get('mode') == 'cascading_overload':
            info_text = (
                f"CASCADING OVERLOAD SIMULATION\n"
                f"----------------------------------------\n"
                f"Cascade Trigger: 1 Primary Chokepoint Failed\n"
                f"Domino Effect: {metrics.get('failed_edges_total', 1) - 1} secondary roads collapsed\n"
                f"Cascade Depth: {metrics.get('cascade_steps', 0)} steps\n"
                f"Network Fragmentation: {metrics.get('capacity_lost_pct', 0):.1f}% capacity lost"
            )
        else:
            info_text = (
                f"RANDOM ATTACK SIMULATION\n"
                f"----------------------------------------\n"
                f"Total Intersections Failed: {metrics.get('failed_nodes', 0)}\n"
                f"New Islands Created: {metrics.get('islands_created', 0)}\n"
                f"Network Fragmentation: {metrics.get('capacity_lost_pct', 0):.1f}% capacity lost"
            )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                     fontsize=12, color='white', verticalalignment='top',
                     bbox=dict(facecolor='black', alpha=0.75, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[*] 2D Visualization saved to: {save_path}")

# ---------------------------------------------------------
# Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print("  NETWORK RESILIENCE & DISASTER SIMULATION (GRAPH THEORY)")
    print("="*70)
    
    img = 5668
    test_img_path = f"archive/train/{img}_sat.jpg"   
    test_mask_path = f"archive/train/{img}_mask.png" 
    
    if os.path.exists(test_img_path) and os.path.exists(test_mask_path):
        print(f"[*] Loading satellite image: {test_img_path}")
        original_img = io.imread(test_img_path)
        img_h, img_w = original_img.shape[:2]
        
        print(f"[*] Loading segmentation mask: {test_mask_path}")
        mask_img = io.imread(test_mask_path, as_gray=True)
        
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
            export_graph = project_graph_to_geo(export_graph, img_width=img_w, img_height=img_h, img_path=test_img_path)
            
            os.makedirs("resilience", exist_ok=True)
            geojson_path = f"resilience/Network_Resilience_3D_{img}.geojson"
            print("\n[*] 5. Exporting GeoJSON for Kepler.gl...")
            export_to_geojson(export_graph, geojson_path)
            
            print("\n[*] 6. Generating 2D Geospatial Visualization Layer...")
            save_file = f"resilience/Network_Resilience_Analysis_{img}.png"
            visualize_network_resilience(original_img, mask_img, export_graph, metrics=cascade_metrics, save_path=save_file)
            
            print("\n[+] Graph Resilience Analysis & Geo Export Complete!")
        else:
            print("[!] Warning: No valid road network found in the mask to build a graph.")
    else:
        print(f"[!] Example files not found.")
