import streamlit as st
import folium
import geopandas as gpd
import branca.colormap as cm
import tempfile
import ast, json, math

from utils.get_closest_indices import KDTreeCache
import utils.calculate_weighted_score as cw

LANDMARKS_FILE = "data/dangerscore_log.geojson"
POLYGONS_FILE = "data/polygons.geojson"

# ---------- caching: load/build ONCE (unless code/data changes) ----------
@st.cache_data(show_spinner=False)
def load_data():
    landmarks = gpd.read_file(LANDMARKS_FILE).to_crs("EPSG:4326")
    polygons = gpd.read_file(POLYGONS_FILE).to_crs("EPSG:4326")
    return landmarks, polygons

@st.cache_resource(show_spinner=False)
def get_kdtree(landmarks):
    return KDTreeCache(landmarks)

landmarks, polygons = load_data()
kdtree_cache = get_kdtree(landmarks)

colormap = cm.linear.YlOrRd_09.scale(0, 100)

# Danger score calculation
def get_danger_score(lat, lng, hour, month):
    k = 4
    closest_landmarks = kdtree_cache.query([(lat, lng)], k)
    distances, indices, properties = closest_landmarks[0]
    return cw.weighted_score(hour, month, distances, properties)

# ---------- UI ----------
st.title("Danger Score Map")

# Put controls in a FORM so moving sliders does NOT trigger generation logic
with st.sidebar.form("controls", clear_on_submit=False):
    st.header("Input Parameters")
    month = st.slider("Select Month", 1, 12, 8)
    hour = st.slider("Select Hour of Day", 0, 23, 12)
    generate = st.form_submit_button("Generate Map")

# Only runs when the button in the form is pressed
if generate:
    st.write(f"Generating map for month={month}, hour={hour} ...")
    progress = st.progress(0.0)
    status = st.empty()

    m = folium.Map(location=[62.0, 15.0], zoom_start=5, control_scale=True)
    folium.TileLayer("CartoDB positron").add_to(m)

    total = len(polygons)
    for i, (_, row) in enumerate(polygons.iterrows(), start=1):
        # update progress occasionally (keeps app responsive)
        if i % 50 == 0 or i == total:
            progress.progress(i / total)
            status.write(f"Processing polygons: {i}/{total}")

        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if hasattr(geom, "is_valid") and not geom.is_valid:
            continue

        center = geom.centroid
        lat, lng = center.y, center.x

        try:
            danger_score = float(get_danger_score(lat, lng, hour, month))
            danger_score = max(0, min(100, danger_score))
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
            # keep going if one polygon fails
            print(f"Error processing polygon: {e}")

    status.write("Done.")
    progress.progress(1.0)

    # display map
    html = m.get_root().render()
    st.components.v1.html(html, width=700, height=500)
else:
    st.info("Adjust month/hour, then click **Generate Map**.")
