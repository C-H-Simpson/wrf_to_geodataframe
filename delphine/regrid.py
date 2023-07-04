"""
A script for regridding pop data, using an integral preserving method and
a non-rectilinear grid.

Standard tools in Iris and CDO do not seem to be able to do this operation when
the grid is non-rectilinear.

If you attempt to do this with too large a grid, you will run out of memory.

Charles Simpson
2023-07-03

"""
# %%
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
from scipy.spatial import Voronoi

from wrf_voronoi import voronoi_finite_polygons_2d

# Specify the file paths
path_grid = Path(r"delphine/RegriddingPop/tas_All_daymax.nc")
path_pop = Path(r"delphine/RegriddingPop/WorldPop_AGL_TasDomain.nc")
output_path = path_grid.parent / f"{path_pop.stem}_regrid.nc"

# Specify things to do with the destination grid file.
input_grid = dict(
    varname="tas",
    lat_name="lat",
    lon_name="lon",
    proj_x_name="x",
    proj_y_name="y",
    grid_crs="EPSG:4326",  # Check this! I think it's right
)
input_pop = dict(
    varname="Band1",
    lat_name="lat",
    lon_name="lon",
    grid_crs="EPSG:4326",  # Check this! I think it's right
)


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


def get_vor(ds, lat_name, lon_name, proj_x_name, proj_y_name, grid_crs, varname):
    """
    Get finite Voronoi polygons for a non-rectlinear grid from an xarray d[input_grid["varname"]]et.
    """
    # Get coordinates
    ds = ds.sortby([proj_x_name, proj_y_name])
    lat_, lon_ = ds[lat_name].values.ravel(), ds[lon_name].values.ravel()
    x, y = np.meshgrid(ds[proj_x_name].values, ds[proj_y_name].values)
    points = np.column_stack((lon_, lat_))
    # Compute Voronoi tesselation.
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    # Implicitly the regions are in the same order as the points contained in vor.point_region.
    boxes = [
        (
            shapely.geometry.Polygon(vertices[g])
            if -1 not in g
            else shapely.geometry.Point(0, 0)
        )
        for g in regions
    ]
    boxes = [
        b.intersection(
            shapely.geometry.box(lon_.min(), lat_.min(), lon_.max(), lat_.max())
        )
        for b in boxes
    ]
    # Turn it into a GeoDataFrame
    gdf_vor = gpd.GeoDataFrame(
        geometry=boxes,
        crs=grid_crs,
    )
    # Join it back to the points to make sure the x and y coordinates are correct.
    gdf_points = gpd.GeoDataFrame(
        {
            proj_x_name: x.ravel(),
            proj_y_name: y.ravel(),
            lat_name: lat_.ravel(),
            lon_name: lon_.ravel(),
        },
        geometry=gpd.points_from_xy(lon_, lat_),
        crs=grid_crs,
    )
    gdf_vor = gdf_vor.sjoin(gdf_points)
    gdf_vor = gdf_vor.set_index([proj_y_name, proj_x_name])
    return gdf_vor


def get_vor_rectilinear(ds, lat_name, lon_name, grid_crs, varname):
    """
    Construct a rectilinear grid from an xarray object.
    """
    # Get coordinates
    lat_, lon_ = ds[lat_name].values, ds[lon_name].values
    width, height = (lon_[1] - lon_[0]) / 2, (lat_[1] - lat_[0]) / 2
    lat_, lon_ = np.meshgrid(lat_, lon_)
    lat_, lon_ = lat_.ravel(), lon_.ravel()
    # In this case, we already know exactly what the Voronoi tesselation is, as
    # the grid is rectilinear. This saves a bit of compute time.
    boxes = [
        shapely.geometry.box(x1 - width, y1 - height, x1 + width, y1 + height)
        for x1, y1 in zip(lon_, lat_)
    ]
    gdf_vor = gpd.GeoDataFrame(
        {lat_name: lat_, lon_name: lon_},
        geometry=boxes,
        crs=grid_crs,
    )
    gdf_vor = gdf_vor.set_index([lat_name, lon_name])
    return gdf_vor


# %%
# Get the gridding of the air temperature.
ds_grid = xr.open_dataset(path_grid).isel(time=1)
gdf_grid = get_vor(ds_grid, **input_grid)
# # %%
# # Visually check the input
# ax = plt.axes(
#     projection=ccrs.Orthographic(
#         central_longitude=ds_grid.lon.mean().item(),
#         central_latitude=ds_grid.lat.mean().item(),
#     )
# )
# ds_grid[input_grid["varname"]].plot(ax=ax)
# ax.gridlines(
#     crs=ccrs.PlateCarree(),
#     draw_labels=True,
#     linewidth=2,
#     color="gray",
#     alpha=0.5,
#     linestyle="--",
# )
# # %%
# # Visually check the output
# ax = plt.axes(
#     projection=ccrs.Orthographic(
#         central_longitude=ds_grid.lon.mean().item(),
#         central_latitude=ds_grid.lat.mean().item(),
#     )
# )
# gdf_grid.join(
#     ds_grid[input_grid["varname"]].to_dataframe()[input_grid["varname"]]
# ).plot(input_grid["varname"], ax=ax, edgecolor="none")
# ax.gridlines(
#     crs=ccrs.PlateCarree(),
#     draw_labels=True,
#     linewidth=2,
#     color="gray",
#     alpha=0.5,
#     linestyle="--",
# )

# %%
# Get the gridding of the population. (This takes a few minutes)
ds_pop = xr.open_dataset(path_pop)
gdf_pop = get_vor_rectilinear(ds_pop, **input_pop)
gdf_pop = gdf_pop.join(
    ds_pop.to_dataframe()[input_pop["varname"]]
)  # Assign the population to the new geodataframe.
# # %%
# # Check it visually.
# ds_pop[input_pop["varname"]].plot()
# gdf_pop[gdf_pop[input_pop["varname"]]>0].plot(input_pop["varname"], edgecolor="none")

# %%
gdf_pop = gdf_pop.fillna(0)  # don't want NaN population
gdf_pop = gdf_pop.assign(area_pop=gdf_pop.area)
gdf_grid = gdf_grid.assign(area_grid=gdf_grid.area)

# %%
# Get a set of geometries that encodes the intersection of the two sets of
# geometries. (This takes a few minutes)
gdf_join = gpd.overlay(
    gdf_pop, gdf_grid, "intersection"
)  # This automatically uses a spatial index, but will still take a while as there are a lot of geometries.
gdf_join = gdf_join.assign(area_inter=gdf_join.area)

# %%
# Now find the intersection between the two, and use area weighting to put the
# population data into the gridding of the temperature data.
#
# index_right is the index of gdf_grid, which came in when the points were
# spatially joined to the Voronoi polygons.
#
# (This takes a few minutes.)
df_result = gdf_join.groupby("index_right").apply(
    lambda _df: ((_df[input_pop["varname"]] / _df.area_pop) * _df.area_inter).sum()
)
# %%
# Assign the result to the grid.
gdf_grid = (
    gdf_grid.set_index("index_right")
    .assign(population=df_result)
    .set_index(gdf_grid.index)
)

# # %%
# # Visually check the ouput
# ax = plt.axes(
#     projection=ccrs.Orthographic(
#         central_longitude=ds_pop.lon.mean().item(),
#         central_latitude=ds_pop.lat.mean().item(),
#     )
# )
# gdf_grid.plot("population", ax=ax)
# ax.gridlines(
#     crs=ccrs.PlateCarree(),
#     draw_labels=True,
#     linewidth=2,
#     color="gray",
#     alpha=0.5,
#     linestyle="--",
# )

# %%
# Turn the result back into an xarray object.
ds_grid_pop = gdf_grid.to_xarray()[["population", "lat", "lon"]].astype("float32")

# %%
# Visually check that the result makes sense.
# Before
ds_pop[input_pop["varname"]].where(ds_pop[input_pop["varname"]] > 1).plot()
# %%
# After
ds_grid_pop.population.where(ds_grid_pop.population > 1).plot()
# Fewer grid cells have identically zero population, many have small values.

# %%
# Check that the total population is close to conserved.
print(
    "Checking conservation of population. Original population / new population =",
    ds_pop[input_pop["varname"]].sum().item() / ds_grid_pop.population.sum().item(),
)

# %%
ds_grid_pop[["population", "lat", "lon"]].to_netcdf(output_path)

# %%
# Check that it is possible to calculate a population weighted temperature.
(ds_grid_pop.population * ds_grid.tas).sum() / (ds_grid_pop.population.sum())

# %%
output_path

# %%
