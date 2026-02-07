import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from typing import List, Tuple

import ast
import json
import math


def _parse_weights(v):
    """Parse weights if needed. Accept dict as-is; parse stringified JSON or Python literal."""
    if v is None:
        return None

    # NaN guard (if pandas gives float NaN)
    try:
        if isinstance(v, float) and math.isnan(v):
            return None
    except Exception:
        pass

    # Already parsed
    if isinstance(v, dict):
        return v

    # Sometimes it's a string
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None

        # Try JSON first
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # Fallback: Python literal (single quotes, etc.)
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    # Unknown type
    return None


class KDTreeCache:
    def __init__(self, landmarks: gpd.GeoDataFrame, crs_projected: str = "EPSG:3006"):
        self.landmarks = landmarks.to_crs(crs_projected)
        self.kd_tree = cKDTree(list(zip(self.landmarks.geometry.x, self.landmarks.geometry.y)))

    def query(
        self,
        points: List[Tuple[float, float]],
        k: int = 3,
        crs_latlon: str = "EPSG:4326",
    ) -> List[Tuple[List[float], List[int], List[dict]]]:

        points_gdf = gpd.GeoDataFrame(
            geometry=[Point(lng, lat) for lat, lng in points],
            crs=crs_latlon,
        ).to_crs(self.landmarks.crs)

        points_coords = list(zip(points_gdf.geometry.x, points_gdf.geometry.y))

        distances, indices = self.kd_tree.query(points_coords, k=k)

        closest_properties = []
        for idx_list in indices:
            properties = [self.landmarks.iloc[i].to_dict() for i in idx_list]

            for prop in properties:
                if "weights" in prop:
                    prop["weights"] = _parse_weights(prop["weights"])

                    # Optional: if weights failed to parse, make it explicit
                    # so downstream code can handle it cleanly.
                    if prop["weights"] is None:
                        prop["weights"] = {"hourly_weights": [0.0]*24, "monthly_weights": [0.0]*12}

            closest_properties.append(properties)

        # If you truly want 1-based indices:
        indices_out = (indices + 1).tolist()

        return list(zip(distances.tolist(), indices_out, closest_properties))
