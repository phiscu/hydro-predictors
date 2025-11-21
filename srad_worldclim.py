import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt

# --- user inputs ---
tif_dir   = Path("/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/srad/wc2.1_30s_srad/issykkul")
shp_path  = Path("/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/catchments/issyk_kul/shp/issyk_kul_outline_exLake_hydrosheds.shp")
out_nc    = Path("/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/srad/srad_issykkul.nc")

# --- load AOI & prep geometry ---
aoi = gpd.read_file(shp_path)
# dissolve to one multipart geometry (handles multiple polygons)
aoi = aoi.to_crs("EPSG:4326").dissolve()  # WorldClim is geographic (WGS84)
aoi_geom = [aoi.geometry.iloc[0]]

# --- helper to open, clip, and standardize one month ---
def load_clip_month(m):
    tif = tif_dir / f"srad_issykkul_{m:02d}.tif"
    da = rxr.open_rasterio(tif, masked=True)               # shape: (band=1, y, x)
    da = da.squeeze("band", drop=True)                      # -> (y, x)
    # ensure CRS is set
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326", inplace=False)
    # clip to AOI (both must share CRS)
    clipped = da.rio.clip(aoi_geom, aoi.crs, drop=True)
    clipped = clipped.rename("srad")
    clipped = clipped.assign_coords(month=m).expand_dims("month")
    return clipped

# --- stack 12 months ---
monthly = xr.concat([load_clip_month(m) for m in range(1, 13)], dim="month")

# --- add minimal metadata ---
monthly = monthly.assign_attrs(
    long_name="Mean daily solar radiation (monthly climatology)",
    source="WorldClim v2.1 SRAD (30 arc-sec)",
    units="kJ m-2 day-1",
    comment="Months are 1..12 for Jan..Dec; values are long-term monthly means."
)
monthly["month"].attrs["long_name"] = "calendar month (1=Jan)"

# --- (optional) create annual mean ---
annual_mean = monthly.mean("month")
annual_mean = annual_mean.assign_attrs(
    long_name="Annual mean of mean daily solar radiation",
)

# --- save to NetCDF ---
monthly.to_netcdf(
    out_nc,
    engine="netcdf4",
    encoding={
        "srad": {
            "zlib": True,
            "complevel": 4,
            "_FillValue": float(monthly.rio.nodata)
        }
    }
)

print(monthly)
print("Wrote:", out_nc)


## Example: plot the July (month=7) SRAD map
monthly.sel(month=7).plot(
    cmap="YlOrRd",
    figsize=(8,6),
    cbar_kwargs={"label": "SRAD [kJ m$^{-2}$ day$^{-1}$]"}
)
plt.title("WorldClim v2.1 SRAD – July")
plt.show()