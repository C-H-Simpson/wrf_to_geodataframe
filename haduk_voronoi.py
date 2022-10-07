"""
Make a geodataframe for a HadUK timestep.
This should be more "portable" than the other example.
NB these grids are for computation not plotting really.

Charles Simpson
2022-10-07
"""
# %%
from wrf_voronoi import voronoi_finite_polygons_2d
import xarray as xr
from scipy.spatial import Voronoi
import shapely
from pathlib import Path
import numpy as np
import geopandas as gpd


# %%
input_path = Path(r"..\data\HadUK\tasmin_hadukgrid_uk_1km_day_20180701-20180731.nc")
destination = input_path.parent / "voronoi.gpkg"
varname = "tasmin"
lat_name = "latitude"
lon_name = "longitude"
proj_x_name = "projection_x_coordinate"
proj_y_name = "projection_y_coordinate"
# regardless of what the "projection coordinates" are,
#  we are assuming that there is a latitude/longitude coord system.
grid_crs = "EPSG:4326"

# %%
# Load WRF output, we will only use the coordinates
ds = xr.open_dataset(input_path)

# %%
# Assume the latitude and longitude are stored in latitude and longitude
lat_, lon_ = ds[lat_name].values.ravel(), ds[lon_name].values.ravel()
x, y = np.meshgrid(ds[proj_x_name].values, ds[proj_y_name].values)
points = np.column_stack((lon_, lat_))

# %%
# Compute Voronoi tesselation.
vor = Voronoi(points)
regions, vertices = voronoi_finite_polygons_2d(vor)

# %%
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
    {proj_x_name: x.ravel(), proj_y_name: y.ravel()},
    geometry=boxes,
    crs=grid_crs,
)
gdf_vor = gdf_vor[
    gdf_vor.within(
        # Drop the faraway limits of the tesselation.
        shapely.geometry.box(lon_.min(), lat_.min(), lon_.max(), lat_.max())
    )
]

# %%
# Check the grid visually.
# You will not be able to do this if there are too many points, drawing like
# this is much less efficient than drawing a grid like with xarray!
if len(gdf_vor) < 10_000:
    gdf_vor.set_index([proj_x_name, proj_y_name]).join(
        ds.isel(time=1).to_dataframe()[[varname]]
    ).plot(varname)

# %%
# Save the result.
gdf_vor.to_file(destination, index=False)
