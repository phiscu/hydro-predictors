#!/usr/bin/env python3
import re
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt

# --- user inputs ---
tif_dir = Path("/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/CHELSA/rsds/monthly_raw")
out_nc  = Path("/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/CHELSA/rsds/chelsa_rsds_monthly_clim_1979_2019.nc")

target_dtype = "float32"        # continuous variable
preferred_fill = -9999.0        # for writing to NetCDF (CF)

# ---- discover files & build a time index ----
# Expect filenames like: CHELSA_rsds_1979_01_V.2.1.tif (month has leading zero)
pat = re.compile(r".*CHELSA_rsds_(\d{4})_(\d{2})_V\..*\.tif$", re.IGNORECASE)
files = sorted([p for p in tif_dir.glob("*.tif") if pat.match(p.name)])

if not files:
    raise SystemExit(f"No .tif files found in {tif_dir}")

records = []
for p in files:
    m = pat.match(p.name)
    if not m:
        continue
    year = int(m.group(1))
    month = int(m.group(2))
    # Represent each file as the mid-month date (any day works; we just need ordering)
    time = pd.Timestamp(year=year, month=month, day=15)
    records.append((time, p))

if not records:
    raise SystemExit("No files matched expected CHELSA_rsds_YYYY_MM*.tif pattern.")

df = pd.DataFrame(records, columns=["time", "path"]).sort_values("time").reset_index(drop=True)
print(f"Found {len(df)} monthly files from {df.time.min().date()} to {df.time.max().date()}.")

# ---- open stack; squeeze band; handle nodata & scale/offset; ensure consistent grid/CRS ----
das = []
crs_str = None
shape = None

for _, row in df.iterrows():
    # masked=True: rioxarray will mask known nodata (when provided by the raster)
    da = rxr.open_rasterio(row.path, masked=True)   # (band,y,x)
    da = da.squeeze("band", drop=True)              # (y,x)

    # If dtype isn't float, move to float so NaN is representable
    if not np.issubdtype(da.dtype, np.floating):
        da = da.astype("float32")

    # Ensure nodata are NaN in memory:
    nodata = da.rio.nodata
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        da = da.where(da != nodata)

    # Also guard against common sentinel leftovers
    for sentinel in (-9999, -9999.0, 65535, 32767, 3.4028235e38):
        da = da.where(da != sentinel)

    # --- apply scale/offset if present (unpack from packed GeoTIFF) ---
    sf = float(da.attrs.get("scale_factor", 1.0))
    ao = float(da.attrs.get("add_offset", 0.0))
    if sf != 1.0 or ao != 0.0:
        da = da * sf + ao
        # prevent double-application downstream
        da.attrs.pop("scale_factor", None)
        da.attrs.pop("add_offset", None)

    # keep dtype uniform
    if target_dtype:
        da = da.astype(target_dtype)

    # Sanity: consistent grid/CRS
    if crs_str is None:
        crs_str = da.rio.crs.to_string() if da.rio.crs else None
        shape = da.shape
    else:
        if da.shape != shape:
            raise RuntimeError(f"Raster grid mismatch for {row.path.name}: {da.shape} != {shape}")
        if (da.rio.crs and crs_str) and (da.rio.crs.to_string() != crs_str):
            raise RuntimeError(f"CRS mismatch for {row.path.name}: {da.rio.crs} vs {crs_str}")

    # Set in-memory nodata as NaN for plotting/analysis
    da.rio.write_nodata(np.nan, inplace=True)

    # Add time coord and collect
    da = da.assign_coords(time=row.time).expand_dims("time")
    das.append(da)

# Build time stack
stack = xr.concat(das, dim="time").rename("rsds")

# --- convert units from MJ m-2 d-1 to W m-2 ---
# 1 MJ m^-2 d^-1 = 1e6 / 86400 W m^-2 ≈ 11.574074074074074
MJ_M2_DAY_TO_W_M2 = 1e6 / 86400.0
stack = stack * MJ_M2_DAY_TO_W_M2
stack.attrs["units"] = "W m-2"

# --- quick stats (now in W m-2) ---
print("Overall min/max (W m-2):", float(stack.min().compute()), float(stack.max().compute()))
print("Winter mean (W m-2):", float(stack.sel(time=stack.time.dt.month.isin([12,1,2])).mean().compute()))
print("Summer mean (W m-2):", float(stack.sel(time=stack.time.dt.month.isin([6,7,8])).mean().compute()))

# ---- monthly climatology: mean for each calendar month across years ----
# These CHELSA tiles are monthly means already → simple mean across years is appropriate.
clim = (
    stack
    .groupby("time.month")
    .mean("time", keep_attrs=True)
    .rename({"month": "month"})
)

# Reorder month dimension 1..12 just in case
clim = clim.sortby("month")
clim = clim.assign_coords(month=np.arange(1, clim.sizes["month"] + 1, dtype="int16"))

# ---- metadata ----
clim = clim.assign_attrs(
    long_name="Surface downwelling shortwave radiation (monthly climatology 1979-2019)",
    comment="Long-term monthly mean across years for each calendar month. Converted from MJ m-2 d-1 to W m-2.",
    crs=crs_str or "",
    source="CHELSA v2.1 RSDS monthly",
    history="Created by make_rsds_monthly_climatology.py; scale/offset unpacked; units converted to W m-2",
    conventions="CF-1.8",
    units="W m-2",
)
clim["month"].attrs.update(long_name="calendar month (1=Jan)", standard_name="month")

# ---- encoding: compression + fill value (no chunking since not using dask) ----
enc = {
    "rsds": {
        "dtype": target_dtype,
        "zlib": True,
        "complevel": 4,
        "_FillValue": preferred_fill,  # concrete on-disk fill; in-memory is NaN
    }
}

# ---- write NetCDF ----
out_nc.parent.mkdir(parents=True, exist_ok=True)
clim.to_netcdf(out_nc, engine="netcdf4", encoding=enc)
print(clim)
print("Wrote:", out_nc)

# ===== Plot: 12-month panel =====
clim = clim.sortby("month")  # ensure order
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 9), constrained_layout=True)

for i, month in enumerate(clim.month.values, start=1):
    ax = axes.flat[i-1]
    da = clim.sel(month=month)
    im = da.plot.imshow(
        ax=ax,
        cmap="YlOrRd",
        add_colorbar=False
    )
    ax.set_title(f"Month {month}", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label="RSDS [W m$^{-2}$]")

plt.suptitle("CHELSA v2.1 RSDS – Monthly Climatology (1979–2019)", fontsize=14)
plt.show()
