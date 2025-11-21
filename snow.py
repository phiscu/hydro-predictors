import pandas as pd
import rasterio
import xarray as xr
import numpy as np
import os
import geopandas as gpd
import rioxarray as rio

## Definitions

def geotiff2xr(file_path):
    """
    Convert a GeoTIFF file into an xarray DataArray with spatial and temporal dimensions.

    Parameters
    ----------
    file_path : str
        Path to the GeoTIFF file.

    Returns
    -------
    xarray.DataArray or None
        DataArray representing the GeoTIFF data, or None if the file does not contain SWE or MASK.
    """
    with rasterio.open(file_path) as src:
        data = src.read()
        transform = src.transform
        crs = src.crs
        height = src.height
        width = src.width
        number_of_days = data.shape[0]
        x_coords = np.linspace(transform.c, transform.c + (width - 1) * transform.a, width)
        y_coords = np.linspace(transform.f, transform.f + (height - 1) * transform.e, height)

        if "SWE" in file_path:
            da = xr.DataArray(data, dims=("day", "y", "x"),
                              coords={"day": range(1, number_of_days + 1), "y": y_coords, "x": x_coords}, name="SWE")
            da.attrs["crs"] = crs
            da.attrs["transform"] = transform
            return da
        elif "MASK" in file_path:
            ma = xr.DataArray(data, dims=("Non_seasonal_snow", "y", "x"),
                              coords={"Non_seasonal_snow": range(1, number_of_days + 1), "y": y_coords, "x": x_coords},
                              name="Non_seasonal_snow")
            ma.attrs["crs"] = crs
            ma.attrs["transform"] = transform
            return ma
        else:
            return None


def select_tif(directory, keyword1, keyword2):
    """
    Select GeoTIFF files in a directory matching specific keywords.

    Parameters
    ----------
    directory : str
        Path to the directory containing GeoTIFF files.
    keyword1 : str
        First keyword to filter files.
    keyword2 : str
        Second keyword to filter files.

    Returns
    -------
    list
        List of file paths matching the specified keywords.
    """
    specific_tif_files = [os.path.join(directory, file) for file in os.listdir(directory)
                          if file.endswith('.tif') and keyword1 in file and keyword2 in file]
    return specific_tif_files


def swe_stack(
        input_dir,
        start_year=1999,
        end_year=2016,
        mask_keep_value=0,  # keep pixels where MASK == 0 (seasonal snow)
        wy_start_month=10,  # water-year start (Oct 1)
        wy_start_day=1,
        out_path=None,  # e.g. "seasonal_swe_1999_2016.nc" or ".zarr"
        var_name="SWE",
        chunks=None,  # e.g. {"time": 31, "y": 512, "x": 512}
        clip_shapefile=None  # path to shapefile for clipping
):
    """
    Build a daily stack of SWE masked to seasonal snow for multiple years.
    Optionally clip to a shapefile polygon.

    Returns
    -------
    xarray.DataArray with dims (time, y, x)
    """
    year_das = []
    crs = transform = None
    x_coords = y_coords = None

    # Load shapefile if clipping
    if clip_shapefile:
        gdf = gpd.read_file(clip_shapefile)
        # We'll assume all files have same CRS, so we don't know it yet
        clip_gdf = gdf  # reproject later if needed
    else:
        clip_gdf = None

    for year in range(start_year, end_year + 1):
        swe_files = select_tif(input_dir, str(year), "SWE")
        mask_files = select_tif(input_dir, str(year), "MASK")

        if not swe_files or not mask_files:
            print(f"Missing files for year {year}. Skipping…")
            continue

        swe_path = swe_files[0]
        mask_path = mask_files[0]

        with rasterio.open(swe_path) as swe_src, rasterio.open(mask_path) as mask_src:
            # Check grid consistency
            if (swe_src.width != mask_src.width) or (swe_src.height != mask_src.height):
                raise ValueError(
                    f"Grid mismatch in {year}: SWE {swe_src.width}x{swe_src.height} vs MASK {mask_src.width}x{mask_src.height}")
            if swe_src.transform != mask_src.transform:
                raise ValueError(f"Transform mismatch in {year}. Reprojection/resampling needed.")

            crs = swe_src.crs
            transform = swe_src.transform
            height, width = swe_src.height, swe_src.width

            x_coords = np.arange(width) * transform.a + transform.c
            y_coords = np.arange(height) * transform.e + transform.f

            swe = swe_src.read()  # (day, y, x)
            mask2d = mask_src.read(1)  # (y, x)

            da_swe = xr.DataArray(
                swe.astype("float32"),
                dims=("day", "y", "x"),
                coords={"day": np.arange(swe.shape[0]), "y": y_coords, "x": x_coords},
                name=var_name
            )
            da_swe.rio.write_crs(crs, inplace=True)

            # Handle SWE nodata
            nodata = (swe_src.nodatavals[0]
                      if swe_src.nodatavals and swe_src.nodatavals[0] is not None
                      else swe_src.nodata)
            if nodata is not None:
                da_swe = da_swe.where(da_swe != nodata)

            da_mask = xr.DataArray(mask2d, dims=("y", "x"), coords={"y": y_coords, "x": x_coords})
            keep = (da_mask == mask_keep_value)
            da_masked = da_swe.where(keep)

            # Clip to shapefile if provided
            if clip_gdf is not None:
                if clip_gdf.crs != crs:
                    clip_gdf = clip_gdf.to_crs(crs)
                da_masked = da_masked.rio.clip(clip_gdf.geometry, clip_gdf.crs, drop=True)

            # Assign time coordinate
            start = pd.Timestamp(year=year, month=wy_start_month, day=wy_start_day)
            time = pd.date_range(start=start, periods=da_masked.sizes["day"], freq="D")
            da_masked = da_masked.assign_coords(time=("day", time)).swap_dims({"day": "time"}).drop_vars("day")

            year_das.append(da_masked)

    if not year_das:
        raise ValueError("No valid years found.")

    stack = xr.concat(year_das, dim="time")
    stack.attrs.update({"crs": str(crs), "transform": tuple(transform)})

    if chunks:
        stack = stack.chunk(chunks)

    if out_path:
        enc = {stack.name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.nan}}
        if out_path.endswith(".nc"):
            stack.to_netcdf(out_path, encoding=enc)
        elif out_path.endswith(".zarr"):
            stack.to_dataset(name=stack.name).to_zarr(out_path, mode="w", encoding=enc)
        else:
            stack.to_netcdf(out_path, encoding=enc)
        print(f"Wrote stack to {out_path}")

    return stack


## Create SWE stack for Issyk-Kul
tif_dir = "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/snow/processed"
shape = "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/catchments/issyk_kul/shp/issyk_kul_outline_exLake_hydrosheds.shp"
snow_nc = "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/snow/snow_issykkul_1999_2016.nc"

stack = swe_stack(tif_dir, clip_shapefile=shape, out_path=snow_nc)




## DEBUG VERSION

import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import xarray as xr
import rioxarray as rio

def select_tif(directory, keyword1, keyword2):
    """Select GeoTIFF files in a directory matching two keywords."""
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(".tif") and keyword1 in file and keyword2 in file
    ]


def swe_stack(
    input_dir,
    start_year=1999,
    end_year=2016,
    mask_keep_value=0,      # keep pixels where MASK == 0
    wy_start_month=10,      # water-year start (Oct 1)
    wy_start_day=1,
    out_path=None,          # e.g. "seasonal_swe_1999_2016.nc" or ".zarr"
    var_name="SWE",
    chunks=None,            # e.g. {"time": 31, "y": 512, "x": 512}
    clip_shapefile=None     # path to shapefile for clipping
):
    """
    Build a daily stack of SWE masked to seasonal snow for multiple years.
    Optionally clip to a shapefile polygon.

    Parameters
    ----------
    input_dir : str
        Directory containing SWE and MASK GeoTIFFs.
    start_year, end_year : int
        Year range to process.
    mask_keep_value : int
        Keep pixels where MASK == this value.
    wy_start_month, wy_start_day : int
        Start date of water year.
    out_path : str or None
        If ends with .zarr, writes incrementally to Zarr.
        If ends with .nc, writes one big NetCDF file.
    var_name : str
        Name of SWE variable.
    chunks : dict or None
        Chunk sizes for dask/xarray.
    clip_shapefile : str or None
        Path to shapefile for clipping.

    Returns
    -------
    xarray.DataArray
    """
    crs = transform = None
    x_coords = y_coords = None
    stack_list = []

    # Load shapefile if clipping
    clip_gdf = None
    if clip_shapefile:
        clip_gdf = gpd.read_file(clip_shapefile)

    for year in range(start_year, end_year + 1):
        swe_files = select_tif(input_dir, str(year), "SWE")
        mask_files = select_tif(input_dir, str(year), "MASK")

        if not swe_files or not mask_files:
            print(f"Missing files for {year}, skipping…")
            continue

        swe_path = swe_files[0]
        mask_path = mask_files[0]

        with rasterio.open(swe_path) as swe_src, rasterio.open(mask_path) as mask_src:
            if (swe_src.width != mask_src.width) or (swe_src.height != mask_src.height):
                raise ValueError(f"Grid mismatch in {year}")
            if swe_src.transform != mask_src.transform:
                raise ValueError(f"Transform mismatch in {year}")

            crs = swe_src.crs
            transform = swe_src.transform
            height, width = swe_src.height, swe_src.width

            x_coords = np.arange(width) * transform.a + transform.c
            y_coords = np.arange(height) * transform.e + transform.f

            swe = swe_src.read().astype("float32")  # (day, y, x)
            mask2d = mask_src.read(1)               # (y, x)

            da_swe = xr.DataArray(
                swe,
                dims=("day", "y", "x"),
                coords={"day": np.arange(swe.shape[0]), "y": y_coords, "x": x_coords},
                name=var_name
            ).rio.write_crs(crs)

            # Apply SWE nodata mask
            nodata = swe_src.nodatavals[0] if swe_src.nodatavals else swe_src.nodata
            if nodata is not None:
                da_swe = da_swe.where(da_swe != nodata)

            # Apply seasonal snow mask
            keep = xr.DataArray(mask2d == mask_keep_value, dims=("y", "x"),
                                coords={"y": y_coords, "x": x_coords})
            da_masked = da_swe.where(keep)

            # Clip if shapefile provided
            if clip_gdf is not None:
                if clip_gdf.crs != crs:
                    clip_gdf = clip_gdf.to_crs(crs)
                da_masked = da_masked.rio.clip(clip_gdf.geometry, clip_gdf.crs, drop=True)

            # Assign time coordinate
            start = pd.Timestamp(year=year, month=wy_start_month, day=wy_start_day)
            time = pd.date_range(start=start, periods=da_masked.sizes["day"], freq="D")
            da_masked = da_masked.assign_coords(time=("day", time)).swap_dims({"day": "time"}).drop_vars("day")

            # Write directly to Zarr if requested
            if out_path and out_path.endswith(".zarr"):
                mode = "w" if year == start_year else "a"
                if mode == "a":
                    da_masked.to_dataset(name=var_name).to_zarr(out_path, mode=mode, append_dim="time")
                else:
                    da_masked.to_dataset(name=var_name).to_zarr(out_path, mode=mode)

            else:
                stack_list.append(da_masked)

    # If not writing incrementally, combine in memory
    if stack_list:
        stack = xr.concat(stack_list, dim="time")
        if chunks:
            stack = stack.chunk(chunks)
        if out_path and out_path.endswith(".nc"):
            enc = {var_name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.nan}}
            stack.to_netcdf(out_path, encoding=enc)
        return stack
    else:
        # If we wrote directly to Zarr, load it back if desired
        if out_path and out_path.endswith(".zarr"):
            return xr.open_zarr(out_path)
        else:
            raise ValueError("No data processed.")


# Create SWE stack for Issyk-Kul
tif_dir = "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/snow/processed"
shape = "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/catchments/issyk_kul/shp/issyk_kul_outline_exLake_hydrosheds.shp"
out_path = "/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/snow/snow_issykkul_1999_2016.zarr"

stack = swe_stack(tif_dir, clip_shapefile=shape, out_path=out_path)
