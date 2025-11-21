import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import gc
import math
import time
from tqdm.auto import tqdm
from rasterio.enums import Resampling

## Load SWE:

# --- Load Zarr version ---
zarr_path = "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/snow/snow_issykkul_1999_2016.zarr"
ds_zarr = xr.open_zarr(zarr_path)

# --- Access the SWE variable ---
swe = ds_zarr["SWE"]

# Calculate daily snow melt and accumulation from SWE
delta = swe.diff('time')

# Valid only where both adjacent days exist
valid = np.isfinite(swe.isel(time=slice(0, -1))) & np.isfinite(swe.isel(time=slice(1, None)))

# Melt: positive mm where SWE decreased
melt = (-delta).where(delta < 0)        # keeps NaN where delta is NaN or >=0
melt = melt.where(valid)                # ensure nodata pairs remain NaN
# If you want 0 for valid non-melt (but keep NaN for nodata), uncomment:
melt = melt.fillna(0).where(valid, np.nan)
melt = melt.rename('melt')


## Load ERA5-Land

# temperature
era5_path = "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/ERA5L/ERA5L_t2m_1999_2018_bbox.nc"

ds = xr.open_dataset(
    era5_path, chunks="auto", mask_and_scale=True
)
t2m_src = ds["t2m"].astype("float32")

# rename to x/y and set CRS (WGS84)
rename_map = {}
for cand, new in (("lon","x"), ("longitude","x"), ("lat","y"), ("latitude","y")):
    if cand in t2m_src.dims:
        rename_map[cand] = new
t2m_src = t2m_src.rename(rename_map)

# Geopontential
geopot_path = "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/ERA5L/ERA5_land_Z_geopotential.nc"
geopot = xr.open_dataset(geopot_path)               # Geopotential height in m^2/s^2
era_elv = (geopot["z"].astype("float32") / 9.80665)   # convert to meters

# Drop non-spatial-coords
era_elv = era_elv.reset_coords(drop=True)

# Rename + set CRS
era_elv = era_elv.rename({"lon":"x","lat":"y"}).rio.write_crs("EPSG:4326", inplace=False)

# drop any non-dim coords
era_elv.reset_coords(drop=True)

# Plot mean temperature over time
# t2m_src.mean(dim="time").plot(robust=True, cmap="coolwarm")
# plt.title("Mean 2m Temperature (ERA5-Land)")
# plt.show()

## Load DEM

# Path to DEM file
dem_path = "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/MERIT_DEM/MERIT30_dem.tif"

dem = rxr.open_rasterio(dem_path, masked=True).squeeze(drop=True)

# Apply CF-style scale/offset if present
sf = float(dem.attrs.get("scale_factor", 1.0) or 1.0)
off = float(dem.attrs.get("add_offset", 0.0) or 0.0)
dem = (dem * sf + off).astype("float32")
dem.name = "elevation"

if dem.rio.crs is None:
    dem = dem.rio.write_crs("EPSG:4326")

## ---------------------------------------------------------------------------------------------------------------------
# Reprojections

# Templates
swe_template  = swe.isel(time=0)
era5_template = (t2m_src.isel(time=0)
                 .rio.set_spatial_dims("x","y")
                 .rio.write_crs("EPSG:4326"))

# --- High-res DEM on SWE grid (area mean) ---
dem_hr = dem.rio.reproject_match(
    swe_template, resampling=Resampling.average, nodata=np.nan
).astype("float32")

# --- ERA5-Land reference elevation from geopotential ---

# Upsample ERA5 orography directly to SWE grid
era5_elv_on_swe = era_elv.rio.reproject_match(
    swe_template, resampling=Resampling.bilinear, nodata=np.nan
).astype("float32")

del dem, era_elv, geopot; gc.collect()  # free memory


## ---------------------------------------------------------------------------------------------------------------------
# Calculate distributed melt rates

lapse_rate = 6.5  # K per km
T = melt.sizes['time']
step = 31  # ~1 month
n_batches = math.ceil(T / step)

num_acc = None
den_acc = None

dz_km = (dem_hr - era5_elv_on_swe) / 1000.0  # elevation difference between DEM cell and ERA5L cell [km]

with rasterio.Env(GDAL_CACHEMAX=128):
    for s in tqdm(range(0, T, step), total=n_batches, unit="batch", desc="DDF monthly batches"):
        e = min(s + step, T)
        times = melt.time.isel(time=slice(s, e))

        tb = (t2m_src.sel(time=times)
              .rio.set_spatial_dims("x","y")
              .rio.write_crs("EPSG:4326")
              .rio.reproject_match(swe.isel(time=0),
                                   resampling=Resampling.bilinear,
                                   nodata=np.nan,
                                   warp_mem_limit=256))

        m = melt.sel(time=times)
        mmask = (m > 0) & np.isfinite(m)

        T_adj = tb - lapse_rate * dz_km
        Tp = xr.where(T_adj > 0, T_adj, 0)

        num_bm = ((m * Tp).where(mmask)).groupby("time.month").sum("time")
        den_bm = (((Tp**2)).where(mmask)).groupby("time.month").sum("time")

        num_bm = num_bm.compute().reindex(month=np.arange(1,13), fill_value=0).astype("float32")
        den_bm = den_bm.compute().reindex(month=np.arange(1,13), fill_value=0).astype("float32")

        num_acc = num_bm if num_acc is None else (num_acc + num_bm)
        den_acc = den_bm if den_acc is None else (den_acc + den_bm)

        del tb, m, mmask, T_adj, Tp, num_bm, den_bm
        gc.collect()

ddf_monthly = (num_acc / den_acc).where(den_acc > 0).astype("float32")
ddf_monthly.name = "ddf_monthly"
ddf_monthly.attrs.update({
    "units": "mm / K / day",
    "description": "Monthly mean degree-day factor aggregated over all years; month=1..12"
})

# Carry the CRS for GIS friendliness
ddf_monthly = ddf_monthly.rio.write_crs(swe.rio.crs, inplace=False)

## Plot
ddf_monthly.sel(month=3).plot(robust=True, cmap="viridis",
    cbar_kwargs={"label": "DDF [mm/K/day]"}); plt.title("Mar DDF"); plt.show()

## Save to NetCDF (compressed)
out_nc = "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/ERA5L/Mean_DDF_500m_monthly_1999_2017.nc"
# Avoid _FillValue clash if present
ddf_monthly.attrs.pop("_FillValue", None)

encoding = {
    "ddf_monthly": {
        "dtype": "float32",
        "zlib": True,
        "shuffle": True,
        "complevel": 5,
        "_FillValue": np.float32(np.nan)
    }
}

ddf_monthly.to_dataset().to_netcdf(out_nc, format="NETCDF4", encoding=encoding)
print(f"Wrote monthly DDF to: {out_nc}")