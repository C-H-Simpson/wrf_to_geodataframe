# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:07:55 2021

Compute the Voronoi polygons for the WRF grid, and save as a geopackage.

@author: C-H-Simpson
"""

import xarray as xr
from scipy.spatial import Voronoi
import shapely
import numpy as np
import geopandas as gpd

from datapaths import wrf_NoUrb_path, wrf_voronoi_path


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    
    Code modified from StackOverflow. 
    https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

if wrf_voronoi_path.is_file() and __name__ != "__main__":
    gdf_vor = gpd.read_file(wrf_voronoi_path)
else:
    # Load WRF output, we will only use the coordinates
    ds_t = xr.open_dataset(wrf_NoUrb_path)
    
    lat_, lon_ = ds_t.XLAT.values.ravel(), ds_t.XLONG.values.ravel()
    x, y = np.meshgrid(ds_t.west_east.values, ds_t.south_north.values)
    points = np.column_stack((lon_, lat_))
    
    # compute Voronoi tesselation
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    
    boxes = [
        (
            shapely.geometry.Polygon(vertices[g])
            if -1 not in g
            else shapely.geometry.Point(0, 0)
        )
        for g in regions
    ]
    gdf_vor = gpd.GeoDataFrame(
        {"west_east": x.ravel(), "south_north": y.ravel()}, geometry=boxes, crs="EPSG:4326"
    )
    gdf_vor = gdf_vor[
        gdf_vor.within(shapely.geometry.box(lon_.min(), lat_.min(), lon_.max(), lat_.max()))
    ]
    
    gdf_vor.to_file(wrf_voronoi_path, driver="GPKG")
    
    del ds_t