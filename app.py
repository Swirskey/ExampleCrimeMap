import streamlit as st
import folium
import geopandas as gpd
from tqdm import tqdm
import os
from utils.get_closest_indices import KDTreeCache
import utils.calculate_weighted_score as cw
import branca.colormap as cm
from streamlit_folium import st_folium
import tempfile

import ast
import json
import math

# File paths
LANDMARKS_FILE = "data/dangerscore_log.geojson"
POLYGONS_FILE = "data/polygons.geojson"

# -----------------------------
# Helpers to fix Streamlit Cloud parsing differences
# -----------------------------
def _parse_maybe_mapping(v):
    """
    Accepts:
      - dict -> returns dict
      - JSON string -> dict/list
      - Python-literal string (single quotes) -> dict/list
      - None/NaN/empty -> None
    """
    if v is None:
        return None

    # Handle NaN (pandas float NaN)
    try:
        if isinstance(v, float) and math.isnan(v):
            return None
    except Exception:
        pass

    if isinstance(v, (dict, list)):
        return v

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None

        # Try JSON first (more portable)
        try:
            return json.loads(s)
        except Exception:
            pass

        # Fallback: Python literal (what causes your error if v is already a dict)
        try:
            return ast.literal_eval(s)
        except Exception:
            return None

    # Unknown type
    return None


def _normalize_landmark_properties(properties):
    """
    properties: list of dicts (one per landmark)
    Returns a new list of dicts with hourly_weights/monthly_weights as real lists if possible.
    """
    norm = []
    for p in properties or []:
        if not isinstance(p, dict):
            norm.append(p)
            continue

        p2 = dict(p)  # shallow copy

        # If your landmarks store weights as a single field (e.g., "weights" dict),
        # normalize it too.
        if "weights" in p2:
            p2["weights"] = _parse_maybe_mapping(p2["weights"])

        # Normalize common fields used for scoring
        for key in ("hourly_weights", "monthly_weights"):
            if key in p2:
                p2[key] = _parse_maybe_mapping(p2[key])

        norm.append(p2)
    return norm


# -----------------------------
# Load GeoJSON files
# -----------------------------
landmarks = gpd.read_file(LANDMARKS_FILE).to_crs("EPSG:4326")
polygons = gpd.read_file(POLYGONS_FILE).to_crs("EPSG:4326")

# KDTree for landmarks
kdtree_cache = KDTreeCache(landmarks)

# Danger score calculation
def get_danger_score(lat, lng, hour, month):
    k = 4
    closest_landmarks = kdtree_cache.query([(lat, lng)], k)
    distances, indices, properties = closest_landmarks[0]

    # ✅ Fix: make sure weights are parsed safely (dict stays dict; strings get parsed)
    properties = _normalize_landmark_properties(properties)

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
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown("### Generating map... Please wait.")
        progress_bar = st.progress(0)

    # Temporary file to save the map
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
        temp_file_path = temp_file.name

        # Initialize Folium map
        m = folium.Map(location=[62.0, 15.0], zoom_start=5, control_scale=True)
        folium.TileLayer("CartoDB positron").add_to(m)

        total = len(polygons)
        processed = 0

        # Add polygons to the map
        with tqdm(total=total, desc="Processing Polygons") as pbar:
            for _, row in polygons.iterrows():
                processed += 1

                # Update Streamlit progress occasionally (keeps UI responsive in Cloud)
                if processed % 25 == 0 or processed == total:
                    progress_bar.progress(min(1.0, processed / total))

                # Skip invalid geometries
                geom = row.get("geometry", None)
                if geom is None or geom.is_empty:
                    pbar.update(1)
                    continue
                if not geom.is_valid:
                    pbar.update(1)
                    continue

                # Get centroid (note: centroid in EPSG:4326 is OK for display purposes)
                center = geom.centroid
                lat, lng = center.y, center.x

                try:
                    danger_score = get_danger_score(lat, lng, hour, month)
                    danger_score = max(0, min(100, float(danger_score)))

                    color = colormap(danger_score)

                    folium.GeoJson(
                        geom,
                        style_function=lambda feature, color=color: {
                            "fillColor": color,
                            "color": color,
                            "weight": 0.5,
                            "fillOpacity": 0.6,
                        },
                        popup=f"Danger Score: {danger_score:.2f}",
                    ).add_to(m)

                except Exception as e:
                    # Show a lightweight error in logs; keep app running
                    print(f"Error processing polygon: {e}")

                pbar.update(1)

        m.save(temp_file_path)

    loading_placeholder.empty()
    st.write(f"Map generated for month: {month}, hour: {hour}")

    with open(temp_file_path, "r", encoding="utf-8") as f:
        map_html = f.read()

    st.components.v1.html(map_html, width=700, height=500)
