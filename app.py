import streamlit as st
import folium
import geopandas as gpd
from tqdm import tqdm
import os
from utils.get_closest_indices import KDTreeCache
import utils.calculate_weighted_score as cw
import branca.colormap as cm
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
    if v is None:
        return None
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
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            return ast.literal_eval(s)
        except Exception:
            return None

    return None


def _normalize_landmark_properties(properties):
    norm = []
    for p in properties or []:
        if not isinstance(p, dict):
            norm.append(p)
            continue

        p2 = dict(p)

        if "weights" in p2:
            p2["weights"] = _parse_maybe_mapping(p2["weights"])

        for key in ("hourly_weights", "monthly_weights"):
            if key in p2:
                p2[key] = _parse_maybe_mapping(p2[key])

        norm.append(p2)
    return norm


# -----------------------------
# Cache expensive loads
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    landmarks = gpd.read_file(LANDMARKS_FILE).to_crs("EPSG:4326")
    polygons = gpd.read_file(POLYGONS_FILE).to_crs("EPSG:4326")
    return landmarks, polygons

@st.cache_resource(show_spinner=False)
def build_kdtree(_landmarks):
    return KDTreeCache(_landmarks)

landmarks, polygons = load_data()
kdtree_cache = build_kdtree(landmarks)

# Define colormap (yellow to red)
colormap = cm.linear.YlOrRd_09.scale(0, 100)

# Danger score calculation
def get_danger_score(lat, lng, hour, month):
    k = 4
    closest_landmarks = kdtree_cache.query([(lat, lng)], k)
    distances, indices, properties = closest_landmarks[0]

    # keep your safety normalization
    properties = _normalize_landmark_properties(properties)

    return cw.weighted_score(hour, month, distances, properties)

# -----------------------------
# Streamlit app layout
# -----------------------------
st.title("Danger Score Map")

st.sidebar.header("Input Parameters")

with st.sidebar.form("controls", clear_on_submit=False):
    month = st.slider("Select Month", 1, 12, 8)
    hour = st.slider("Select Hour of Day", 0, 23, 12)
    generate = st.form_submit_button("Generate Map")

# store last generated map so it stays visible when you move sliders
if "last_map_html" not in st.session_state:
    st.session_state["last_map_html"] = None
if "last_params" not in st.session_state:
    st.session_state["last_params"] = None

if generate:
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown("### Generating map... Please wait.")
        progress_bar = st.progress(0.0)

    # Initialize Folium map
    m = folium.Map(location=[62.0, 15.0], zoom_start=5, control_scale=True)
    folium.TileLayer("CartoDB positron").add_to(m)

    total = len(polygons)
    processed = 0

    # (tqdm is fine for local; on cloud it can be noisy—keeping it since you had it)
    with tqdm(total=total, desc="Processing Polygons") as pbar:
        for _, row in polygons.iterrows():
            processed += 1

            # update Streamlit progress occasionally
            if processed % 50 == 0 or processed == total:
                progress_bar.progress(processed / total)

            geom = row.get("geometry", None)
            if geom is None or geom.is_empty:
                pbar.update(1)
                continue
            if not geom.is_valid:
                pbar.update(1)
                continue

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
                print(f"Error processing polygon: {e}")

            pbar.update(1)

    # Render to HTML and persist
    st.session_state["last_map_html"] = m.get_root().render()
    st.session_state["last_params"] = (month, hour)

    loading_placeholder.empty()
    st.success(f"Map generated for month: {month}, hour: {hour}")


if st.session_state["last_map_html"]:
    mth, hr = st.session_state["last_params"]
    st.caption(f"Showing last generated map — month {mth}, hour {hr}")
    st.components.v1.html(st.session_state["last_map_html"], width=700, height=500)
else:
    st.info("Set month/hour and click **Generate Map**.")
