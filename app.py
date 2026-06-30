import streamlit as st
import subprocess
import time
import os
import glob
import json
import pydeck as pdk

st.set_page_config(page_title="AI Road Topology Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR MINIMALIST NON-NEON LOOK ---
st.markdown("""
<style>
    .reportview-container {
        background: #f8f9fa;
        color: #212529;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #34495e;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2c3e50;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #dee2e6;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("Configuration")
st.sidebar.markdown("Configure the neural network and the image source.")

# Model Selection
model_choice = st.sidebar.selectbox("Select Model Weight", ["mumbai_finetuned_model.pth", "deepglobe_finetuned_model.pth"])

# Image Source Selection
image_source = st.sidebar.radio("Image Source", ["Fetch Random Image (Matches Model)", "Upload Custom Image"])
uploaded_file = None
if image_source == "Upload Custom Image":
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["tif", "png", "jpg"])

st.sidebar.markdown("---")
st.sidebar.subheader("Resilience Settings")
attack_type = st.sidebar.radio("Attack Simulation Type", ["Targeted (Choke Points)", "Random (Chance)"])
removal_pct = st.sidebar.slider("Nodes to Remove (%) [Random Attack Only]", 5, 50, 15)

# --- MAIN AREA ---
st.title("Geospatial Network Resilience Dashboard")
st.markdown("Analyze satellite imagery, extract topology-preserving road networks, and simulate cascading infrastructure failures.")

if st.button("Start Analysis Pipeline"):
    # 1. Setup the terminal expander
    terminal_expander = st.expander("Terminal Execution Logs", expanded=True)
    with terminal_expander:
        log_container = st.empty()
    
    logs = ""
    log_container.code(logs, language="bash")
    
    # 2. Prepare Arguments for the backend pipeline
    import sys
    attack_arg = "targeted" if "Targeted" in attack_type else "random"
    
    # We use "-u" to force python into unbuffered mode so stdout streams live!
    cmd = [sys.executable, "-u", "graph_resilience_spacenet.py", "--model", model_choice, "--attack", attack_arg, "--remove_pct", str(removal_pct/100.0)]
    
    # Save uploaded file temporarily if provided
    if image_source == "Upload Custom Image" and uploaded_file is not None:
        temp_img_path = "temp_uploaded_img" + os.path.splitext(uploaded_file.name)[1]
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        cmd.extend(["--image", temp_img_path])
        
    if image_source == "Upload Custom Image" and uploaded_file is None:
        st.error("Please upload an image first, or select 'Fetch Random'.")
        st.stop()
    
    # We use Popen to stream the output live
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    for line in process.stdout:
        logs += line
        # Update the Streamlit UI with the new log lines
        log_container.code(logs, language="bash")
    
    process.wait()
    
    if process.returncode == 0:
        st.success("Pipeline Execution Complete!")
        terminal_expander.expanded = False # Optionally close it after success
        
        # 3. Display Step-by-Step Results
        st.header("Step-by-Step Visualizations")
        
        col1, col2 = st.columns(2)
        
        # The script saves spacenet_generalize_img.jpg and spacenet_generalize_mask.png
        with col1:
            st.subheader("1. Input Satellite Image")
            if os.path.exists("resilience/spacenet_generalize_img.jpg"):
                st.image("resilience/spacenet_generalize_img.jpg", use_container_width=True)
            
            st.subheader("3. Graph Analysis & Vulnerabilities")
            # Find the most recently generated resilience PNG
            res_pngs = glob.glob("resilience/*.png")
            # Exclude the mask image from the glob search
            res_pngs = [p for p in res_pngs if "mask" not in p]
            if res_pngs:
                latest_png = max(res_pngs, key=os.path.getctime)
                st.image(latest_png, use_container_width=True)
                
        with col2:
            st.subheader("2. AI Extracted Mask (UNet++)")
            if os.path.exists("resilience/spacenet_generalize_mask.png"):
                st.image("resilience/spacenet_generalize_mask.png", use_container_width=True)
                
            st.subheader("4. Threats & Results Summary")
            st.info(f"**Simulation:** {attack_type}")
            # Parse final metrics from logs
            import re
            cap_lost = re.search(r"Network Fragmentation: ([\d.]+)%", logs)
            
            if "Targeted" in attack_type:
                cascade = re.search(r"Cascade Depth: (\d+)", logs)
                failed_edges = re.search(r"Total Roads Failed: (\d+)", logs)
                if cascade and cap_lost and failed_edges:
                    st.metric("Total Network Capacity Lost", f"{cap_lost.group(1)}%")
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("Cascading Steps", cascade.group(1))
                    col_m2.metric("Roads Destroyed", failed_edges.group(1))
            else:
                islands = re.search(r"Islands Created: (\d+)", logs)
                failed_nodes = re.search(r"Total Nodes Failed: (\d+)", logs)
                if islands and cap_lost and failed_nodes:
                    st.metric("Total Network Capacity Lost", f"{cap_lost.group(1)}%")
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("Isolated Neighborhoods", islands.group(1))
                    col_m2.metric("Nodes Destroyed", failed_nodes.group(1))

        
        # 4. Display the PyDeck (Deck.gl) Interactive Map
        st.header("Interactive 3D Network Map")
        st.markdown("Explore the extracted graph nodes and choke points projected onto Earth coordinates.")
        
        # Find the latest geojson
        geojsons = glob.glob("resilience/*.geojson")
        if geojsons:
            latest_geojson = max(geojsons, key=os.path.getctime)
            
            with open(latest_geojson, 'r') as f:
                geo_data = json.load(f)
                
            # PyDeck configuration
            # The geojson has "properties" like "centrality", "color"
            # Pydeck GeoJsonLayer can use these.
            
            layer = pdk.Layer(
                "GeoJsonLayer",
                geo_data,
                pickable=True,
                stroked=False,
                filled=True,
                extruded=True,
                wireframe=True,
                get_fill_color="properties.fill",
                get_line_color="properties.fill",
                get_line_width=2,
                line_width_min_pixels=2,
            )
            
            # Extract coordinates for initial view state (defaulting to Mumbai)
            view_state = pdk.ViewState(
                latitude=18.8928, 
                longitude=72.8100,
                zoom=16,
                pitch=45,
                bearing=0
            )
            
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "{name}\nCentrality: {centrality}"},
                map_style=pdk.map_styles.LIGHT # Minimalist light style, not neon dark
            )
            
            st.pydeck_chart(r)
        else:
            st.error("No GeoJSON output found to render the map.")
            
    else:
        st.error("Pipeline failed. Check terminal logs above.")
