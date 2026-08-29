import ast
import json
import math
from typing import List, Tuple

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree


def parse_weights(value):
    """Parse a landmark's ``weights`` cell.

    Depending on the GeoJSON driver the column arrives either already decoded as
    a dict or still as a string, so accept both JSON and Python-literal strings.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except ValueError:
            pass
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
    return None


class KDTreeCache:
    def __init__(self, landmarks: gpd.GeoDataFrame, crs_projected: str = "EPSG:3006"):
        """
        Initializes the KDTreeCache with a GeoDataFrame of landmarks, projecting the landmarks to a
        specific coordinate reference system (CRS) before building the KDTree.
        """
        # Convert landmarks to the specified projected CRS
        self.landmarks = landmarks.to_crs(crs_projected)

        # Create KDTree using the x, y coordinates of the landmarks
        coords = np.column_stack(
            [self.landmarks.geometry.x.to_numpy(), self.landmarks.geometry.y.to_numpy()]
        )
        self.kd_tree = cKDTree(coords)

        # Decode the weights once up front instead of on every query. Scoring then
        # reduces to array indexing over these three tables.
        n = len(self.landmarks)
        self.scores = self.landmarks["score"].to_numpy(dtype=float)
        self.hourly_weights = np.zeros((n, 24), dtype=float)
        self.monthly_weights = np.zeros((n, 12), dtype=float)
        for i, raw in enumerate(self.landmarks["weights"].to_numpy()):
            weights = parse_weights(raw)
            if not weights:
                continue
            self.hourly_weights[i] = weights["hourly_weights"]
            self.monthly_weights[i] = weights["monthly_weights"]

    def query_batch(self, lats, lngs, k: int = 3, crs_latlon: str = "EPSG:4326"):
        """
        Vectorized nearest-landmark lookup for many points at once.

        Reprojects every point in a single pass and runs one KDTree query, rather
        than rebuilding a one-row GeoDataFrame per point. Returns ``(distances,
        indices)`` arrays of shape ``(len(lats), k)``; indices are 0-based and
        index directly into ``scores`` / ``hourly_weights`` / ``monthly_weights``.
        """
        points = gpd.GeoSeries(
            gpd.points_from_xy(lngs, lats), crs=crs_latlon
        ).to_crs(self.landmarks.crs)
        coords = np.column_stack([points.x.to_numpy(), points.y.to_numpy()])
        return self.kd_tree.query(coords, k=k, workers=-1)

    def query(
        self, points: List[Tuple[float, float]], k: int = 3, crs_latlon: str = "EPSG:4326"
    ) -> List[Tuple[List[float], List[int], List[dict]]]:
        """
        Queries the pre-built KDTree for the closest landmarks to each point and returns their distances,
        indices, and properties, including nested dictionary values.
        """
        lats = [lat for lat, _ in points]
        lngs = [lng for _, lng in points]
        distances, indices = self.query_batch(lats, lngs, k=k, crs_latlon=crs_latlon)

        closest_properties = []
        for idx_list in np.atleast_2d(indices):
            properties = []
            for i in idx_list:
                prop = self.landmarks.iloc[i].to_dict()
                if "weights" in prop:
                    prop["weights"] = parse_weights(prop["weights"])
                properties.append(prop)
            closest_properties.append(properties)

        # Adjust indices (1-based index instead of 0-based)
        indices = indices + 1

        # Return distances, indices, and properties as a list of tuples
        return list(zip(distances.tolist(), indices.tolist(), closest_properties))
