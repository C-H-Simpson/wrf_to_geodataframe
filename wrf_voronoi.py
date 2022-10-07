# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:07:55 2021

Compute the Voronoi polygons for the WRF grid, and save as a GIS file.

@author: C-H-Simpson
"""

# %%
import xarray as xr
from scipy.spatial import Voronoi
import shapely
from pathlib import Path
import numpy as np
import geopandas as gpd


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


# %%
if __name__ == "__main__":
    # %%
    wrf_path = Path(
        r"..\data\wrf_output_220726\WRF_Urb_BouLac_T2-V10-U10_20180525-20180831.nc"
    )
    wrf_voronoi_path = Path("wrf_voronoi.gpkg")

    # Load WRF output, we will only use the coordinates
    ds_t = xr.open_dataset(wrf_path)
    print(ds_t.coords)
    if "x" in ds_t:
        ds_t = ds_t.drop(["x", "y"])  # Drop incorrectly assigned x and y

    # Assume the latitude and longitude are stored in latitude and longitude
    lat_, lon_ = ds_t.XLAT.values.ravel(), ds_t.XLONG.values.ravel()
    x, y = np.meshgrid(ds_t.west_east.values, ds_t.south_north.values)
    points = np.column_stack((lon_, lat_))

    # Compute Voronoi tesselation.
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    # Turn the Voronoi tesselation into a GeoDataFrame.
    boxes = [
        (
            shapely.geometry.Polygon(vertices[g])
            if -1 not in g
            else shapely.geometry.Point(0, 0)
        )
        for g in regions
    ]
    gdf_vor = gpd.GeoDataFrame(
        {"west_east": x.ravel(), "south_north": y.ravel()},
        geometry=boxes,
        crs="EPSG:4326",
    )
    gdf_vor = gdf_vor[
        gdf_vor.within(
            # Drop the faraway limits of the tesselation.
            shapely.geometry.box(lon_.min(), lat_.min(), lon_.max(), lat_.max())
        )
    ]
    gdf_vor.to_file(wrf_voronoi_path)

    # %%
    # You can now easily link data from the xarray object to the geodataframe
    # with a pandas join.
    # Here is an example...
    # Get the mean daily minimum temperature, and make it a pandas dataframe.
    df_wrf_min = (
        ds_t.resample(XTIME="1D")
        .min()
        .mean("XTIME")
        .T2.to_dataframe(name="Tmin")[["Tmin"]]
    )
    df_wrf_max = (
        ds_t.resample(XTIME="1D")
        .max()
        .mean("XTIME")
        .T2.to_dataframe(name="Tmax")[["Tmax"]]
    )
    df_wrf_mean = (
        ds_t.resample(XTIME="1D")
        .mean()
        .mean("XTIME")
        .T2.to_dataframe(name="Tmean")[["Tmean"]]
    )
    # Link it to the geometries.
    gdf_wrf = (
        gdf_vor.set_index(["west_east", "south_north"])
        .join(df_wrf_min)
        .join(df_wrf_max)
        .join(df_wrf_mean)
    )
    gdf_wrf.plot("Tmin")

    gdf_wrf.to_file("london_heat_island_wholeDomain.gpkg", driver="GPKG")

    gdf_london = gpd.read_file(
        r"C:\Users\ucbqc38\Documents\GIS\statistical-gis-boundaries-london\ESRI\London_Borough_Excluding_MHW.shp"
    )
    gdf_wrf[gdf_wrf.to_crs("EPSG:27700").intersects(gdf_london.unary_union)].to_file(
        "london_heat_island.gpkg", driver="GPKG"
    )

# %%
