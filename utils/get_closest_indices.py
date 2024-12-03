import os
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from typing import List, Tuple

import ast  # For handling string to dictionary conversion

class KDTreeCache:
    def __init__(self, landmarks: gpd.GeoDataFrame, crs_projected: str = "EPSG:3006"):
        """
        Initializes the KDTreeCache with a GeoDataFrame of landmarks, projecting the landmarks to a
        specific coordinate reference system (CRS) before building the KDTree.
        """
        # Convert landmarks to the specified projected CRS
        self.landmarks = landmarks.to_crs(crs_projected)
        
        # Create KDTree using the x, y coordinates of the landmarks
        self.kd_tree = cKDTree(list(zip(self.landmarks.geometry.x, self.landmarks.geometry.y)))

    def query(self, points: List[Tuple[float, float]], k: int = 3, crs_latlon: str = "EPSG:4326") -> List[Tuple[List[float], List[int], List[dict]]]:
        """
        Queries the pre-built KDTree for the closest landmarks to each point and returns their distances,
        indices, and properties, including nested dictionary values.
        """ 
        # Convert the input points (longitude, latitude) to GeoDataFrame with the specified CRS
        points_gdf = gpd.GeoDataFrame(geometry=[Point(lng, lat) for lat, lng in points], crs=crs_latlon)
        
        # Convert points to the same CRS as the landmarks
        points_gdf = points_gdf.to_crs(self.landmarks.crs)

        # Extract coordinates from the points GeoDataFrame
        points_coords = list(zip(points_gdf.geometry.x, points_gdf.geometry.y))

        # Query the KDTree for the nearest 'k' landmarks to each point
        distances, indices = self.kd_tree.query(points_coords, k=k)
        
        # Adjust indices (1-based index instead of 0-based)
        
        # !!!this can surely be done in a better way, but the current formats we use are not optimal.
        closest_properties = []
        for idx_list in indices:
           
            properties = [self.landmarks.iloc[i].to_dict() for i in idx_list]
            for prop in properties:
               
                if 'weights' in prop:
                  
                    prop['weights'] = ast.literal_eval(prop['weights'])


            closest_properties.append(properties)

        indices = indices + 1 
        
        # Return distances, indices, and properties as a list of tuples
        return list(zip(distances.tolist(), indices.tolist(), closest_properties))
