import json

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import streamlit as st

import utils.calculate_weighted_score as cw
from utils.get_closest_indices import KDTreeCache

# File paths
LANDMARKS_FILE = "data/dangerscore_log.geojson"
POLYGONS_FILE = "data/polygons.geojson"

WGS84 = "EPSG:4326"
METRIC_CRS = "EPSG:3006"  # SWEREF99 TM, metres

# The source polygons carry 2.26M vertices (~93 MB of GeoJSON), far more detail
# than a screen can show: even zoomed in to a city, 25 m is under a pixel.
# Simplifying to this tolerance drops ~88% of the vertices and is the single
# biggest win here. Raise it to shrink the payload further (100 m -> ~3.5 MB) at
# the cost of visibly angular borders when zoomed right in.
SIMPLIFY_TOLERANCE_M = 25
COORDINATE_DECIMALS = 5  # ~1 m, well below the simplification tolerance

NEIGHBOURS = 4


# -----------------------------
# Cached loading and preprocessing
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_landmarks():
    return gpd.read_file(LANDMARKS_FILE).to_crs(WGS84)


@st.cache_resource(show_spinner=False)
def build_kdtree():
    return KDTreeCache(load_landmarks())


def _round_coordinates(value):
    if isinstance(value, list):
        return [_round_coordinates(item) for item in value]
    return round(value, COORDINATE_DECIMALS)


@st.cache_resource(show_spinner=False)
def load_polygon_geometry():
    """Load the polygons and prepare everything that does not depend on hour/month.

    Returns the centroid latitudes/longitudes used for scoring (taken from the
    full-resolution geometry) and the simplified GeoJSON geometry dicts used for
    drawing. The returned dicts are shared across reruns and must not be mutated.
    """
    polygons = gpd.read_file(POLYGONS_FILE).to_crs(WGS84)

    geometry = polygons.geometry
    keep = geometry.notna() & ~geometry.is_empty & geometry.is_valid
    geometry = geometry[keep]

    # Score against the true centroids, before any simplification.
    centroids = geometry.centroid
    lats = centroids.y.to_numpy()
    lngs = centroids.x.to_numpy()

    simplified = geometry.to_crs(METRIC_CRS).simplify(
        SIMPLIFY_TOLERANCE_M, preserve_topology=True
    ).to_crs(WGS84)

    collection = json.loads(gpd.GeoSeries(simplified, crs=WGS84).to_json(drop_id=True))
    geometries = []
    for feature in collection["features"]:
        geom = feature["geometry"]
        geom["coordinates"] = _round_coordinates(geom["coordinates"])
        geometries.append(geom)

    return lats, lngs, geometries


# Each cached map is ~7 MB, so keep only a handful. Streamlit Community Cloud
# gives the whole app about 1 GB and the polygon data already claims a chunk.
@st.cache_data(show_spinner=False, max_entries=6)
def render_map_html(month: int, hour: int) -> str:
    """Build the Leaflet map for one month/hour combination and return its HTML."""
    lats, lngs, geometries = load_polygon_geometry()
    kdtree_cache = build_kdtree()
    colormap = cm.linear.YlOrRd_09.scale(0, 100)

    distances, indices = kdtree_cache.query_batch(lats, lngs, k=NEIGHBOURS)
    scores = cw.weighted_score_batch(
        hour,
        month,
        distances,
        indices,
        kdtree_cache.scores,
        kdtree_cache.hourly_weights,
        kdtree_cache.monthly_weights,
    )
    scores = np.clip(np.nan_to_num(scores), 0, 100)

    # One GeoJson layer for every polygon instead of one layer each: the browser
    # builds a single Leaflet object rather than a few thousand.
    features = [
        {
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "danger_score": f"{score:.2f}",
                "color": colormap(score),
            },
        }
        for geometry, score in zip(geometries, scores)
    ]

    m = folium.Map(location=[62.0, 15.0], zoom_start=5, control_scale=True)
    folium.TileLayer("CartoDB positron").add_to(m)
    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        style_function=lambda feature: {
            "fillColor": feature["properties"]["color"],
            "color": feature["properties"]["color"],
            "weight": 0.5,
            "fillOpacity": 0.6,
        },
        popup=folium.GeoJsonPopup(fields=["danger_score"], aliases=["Danger Score:"]),
    ).add_to(m)

    return m.get_root().render()


# -----------------------------
# Streamlit app layout
# -----------------------------
st.title("Danger Score Map")

st.sidebar.header("Input Parameters")

with st.sidebar.form("controls", clear_on_submit=False):
    month = st.slider("Select Month", 1, 12, 8)
    hour = st.slider("Select Hour of Day", 0, 23, 12)
    generate = st.form_submit_button("Generate Map")

if "last_params" not in st.session_state:
    st.session_state["last_params"] = None

if generate:
    with st.spinner("Generating map..."):
        render_map_html(month, hour)
    st.session_state["last_params"] = (month, hour)
    st.success(f"Map generated for month: {month}, hour: {hour}")

if st.session_state["last_params"]:
    last_month, last_hour = st.session_state["last_params"]
    st.caption(f"Showing last generated map — month {last_month}, hour {last_hour}")
    st.components.v1.html(render_map_html(last_month, last_hour), width=700, height=500)
else:
    st.info("Set month/hour and click **Generate Map**.")
