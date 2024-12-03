import streamlit as st
import folium
import geopandas as gpd
from tqdm import tqdm
import os
from utils.get_closest_indices import KDTreeCache
import utils.calculate_weighted_score as cw
import branca.colormap as cm
import random
from streamlit_folium import st_folium
import tempfile

# File paths
LANDMARKS_FILE = "data/dangerscore_log.geojson"
POLYGONS_FILE = "data/polygons.geojson"

# Load GeoJSON files
landmarks = gpd.read_file(LANDMARKS_FILE).to_crs("EPSG:4326")
polygons = gpd.read_file(POLYGONS_FILE).to_crs("EPSG:4326")

# KDTree for landmarks
kdtree_cache = KDTreeCache(landmarks)

# Danger score calculation
def get_danger_score(lat, lng, hour, month):
    k = 4
    closest_landmarks = kdtree_cache.query([(lat, lng)], k)
    distances, indices, properties = closest_landmarks[0]
    return cw.weighted_score(hour, month, distances, properties)

# Define colormap (yellow to red)
colormap = cm.linear.YlOrRd_09.scale(0, 100)

# Streamlit app layout
st.title("Danger Score Map")
st.sidebar.header("Input Parameters")
month = st.sidebar.slider("Select Month", 1, 12, 8)
hour = st.sidebar.slider("Select Hour of Day", 0, 23, 12)

# Button to generate map
if st.sidebar.button("Generate Map"):
    # Display a placeholder for the loading screen
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown("### Generating map... Please wait.")
        st.progress(0)  # Progress bar

    # Temporary file to save the map
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
        temp_file_path = temp_file.name

        # Initialize Folium map
        m = folium.Map(location=[62.0, 15.0], zoom_start=5, control_scale=True)
        folium.TileLayer("CartoDB positron").add_to(m)

        # Add polygons to the map
        with tqdm(total=len(polygons), desc="Processing Polygons") as pbar:
            for _, row in polygons.iterrows():
                # Skip invalid geometries
                if not row["geometry"].is_valid:
                    continue

                # Get the polygon centroid
                center = row["geometry"].centroid
                lat, lng = center.y, center.x

                try:
                    # Calculate danger score
                    danger_score = get_danger_score(lat, lng, hour, month)
                    danger_score = max(0, min(100, danger_score))  # Clamp to colormap range

                    # Style based on danger score
                    color = colormap(danger_score)

                    # Add polygon to map with a specific style function
                    folium.GeoJson(
                        row["geometry"],
                        style_function=lambda feature, color=color: {
                            "fillColor": color,
                            "color": color,
                            "weight": 0.5,
                            "fillOpacity": 0.6,
                        },
                        popup=f"Danger Score: {danger_score:.2f}",
                    ).add_to(m)
                except Exception as e:
                    print(f"Error processing polygon: {e}")

                pbar.update(1)

        # Save the map to the temporary file
        m.save(temp_file_path)

    # Replace loading screen with the generated map
    loading_placeholder.empty()
    st.write(f"Map generated for month: {month}, hour: {hour}")

    # Display the map in Streamlit using the saved file
    with open(temp_file_path, "r") as f:
        map_html = f.read()
    st.components.v1.html(map_html, width=700, height=500)
