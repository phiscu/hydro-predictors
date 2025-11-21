import os
import pyproj
import ee
import geopandas as gpd
import geemap
import requests
import glob
import zipfile
import rasterio
import xarray as xr
import requests
from datetime import date
import rioxarray
import xee
from rasterio.merge import merge
from rasterio.mask import mask
from dask.diagnostics import ProgressBar

# To avoid PROJ error:
os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()

# Initialize Earth Engine
ee.Initialize(project="matilda-edu")

# 1. Load local catchments from a GeoPackage and convert to EE FeatureCollection
gdf = gpd.read_file('/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/issykul_vectors.gpkg', layer='catchment_new')
catchments = geemap.geopandas_to_ee(gdf)

# Define regions
def get_region_catch():
    return catchments.geometry()

def get_region_bbox():
    return catchments.geometry().bounds()

region_catch = get_region_catch()
region_bbox = get_region_bbox()

# Download helper (EE getDownloadURL)
def download_image(image, region, scale, crs, filename):
    params = {
        'name': filename,
        'scale': scale,
        'crs': crs,
        'region': region,
        'filePerBand': False,
    }
    url = image.getDownloadURL(params)
    print(f"Downloading {filename}: {url}")
    resp = requests.get(url, stream=True)
    out_zip = f"{filename}.zip"
    with open(out_zip, 'wb') as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
    print(f"Saved {out_zip}")
    return out_zip

## 1. MODIS ET (8-day, 500m) – YEARLY STACKS
def download_modis_et():
    modis = ee.ImageCollection('MODIS/061/MOD16A2GF')
    def rename_et(img):
        return img.select('ET').rename(
            ee.Date(img.get('system:time_start')).format('YYYY_MM_dd')
        )
    for yr in range(2000, 2025):
        period = modis.filterDate(f'{yr}-01-01', f'{yr}-12-31')
        stack = period.map(rename_et).map(lambda img: img.clip(catchments)).toBands()
        download_image(stack, region_catch, 500, 'EPSG:4326', f'MOD16A2GF_ET_{yr}')

## 2. HiHydroSoil HSG (250m)
def download_soil_hsg():
    soil = ee.Image('projects/sat-io/open-datasets/HiHydroSoilv2_0/Hydrologic_Soil_Group_250m')
    soil_clipped = soil.clip(catchments)
    download_image(soil_clipped, region_catch, 250, 'EPSG:4326', 'HiHydroSoil_HSG')

## 3. GLC-FCS30D – Tiled Downloads
# Split bounding box into tiles to stay under payload limits
coords = region_bbox['coordinates'][0]
xmin, ymin = coords[0]
xmax, ymax = coords[2]
n_rows, n_cols = 3, 3
x_steps = [xmin + i*(xmax-xmin)/n_cols for i in range(n_cols+1)]
y_steps = [ymin + j*(ymax-ymin)/n_rows for j in range(n_rows+1)]

def download_glc():
    # Annual layers
    glc_ann = ee.ImageCollection('projects/sat-io/open-datasets/GLC-FCS30D/annual')
    bands = ee.Image(glc_ann.first()).bandNames().getInfo()
    for i in range(n_cols):
        for j in range(n_rows):
            tile_geom = ee.Geometry.Rectangle([
                x_steps[i], y_steps[j], x_steps[i+1], y_steps[j+1]
            ])
            tile_region = tile_geom.getInfo()
            for idx, band in enumerate(bands):
                year = 2000 + idx
                img = glc_ann.select(band).mosaic().rename(str(year)).clip(tile_geom)
                fname = f'GLC_annual_{year}_tile_{i}_{j}'
                download_image(img, tile_region, 30, 'EPSG:4326', fname)
    # Five-year composites
    glc5 = ee.ImageCollection('projects/sat-io/open-datasets/GLC-FCS30D/five-years-map')
    periods = ['1985_1990', '1990_1995', '1995_2000']
    bands5 = ee.Image(glc5.first()).bandNames().getInfo()
    for i in range(n_cols):
        for j in range(n_rows):
            tile_geom = ee.Geometry.Rectangle([
                x_steps[i], y_steps[j], x_steps[i+1], y_steps[j+1]
            ])
            tile_region = tile_geom.getInfo()
            for idx, band in enumerate(bands5):
                period = periods[idx]
                img5 = glc5.select(band).mosaic().rename(period).clip(tile_geom)
                fname = f'GLC_5yr_{period}_tile_{i}_{j}'
                download_image(img5, tile_region, 30, 'EPSG:4326', fname)

# Local stitching and clipping
def stitch_and_clip(zip_pattern, out_mosaic, out_clipped, clip_gdf):
    # Find zip files matching pattern
    zip_files = glob.glob(zip_pattern)
    tif_files = []
    for z in zip_files:
        with zipfile.ZipFile(z, 'r') as zf:
            extract_dir = z[:-4]
            zf.extractall(extract_dir)
            tif_files.extend(glob.glob(os.path.join(extract_dir, '*.tif')))
    if not tif_files:
        raise ValueError(f"No TIFFs found for pattern {zip_pattern}")
    # Merge
    srcs = [rasterio.open(t) for t in tif_files]
    mosaic, out_trans = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update({
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': out_trans
    })
    with rasterio.open(out_mosaic, 'w', **meta) as dst:
        dst.write(mosaic)
    # Clip to catchments
    with rasterio.open(out_mosaic) as src:
        out_img, out_tr = mask(src, clip_gdf.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({'height': out_img.shape[1], 'width': out_img.shape[2], 'transform': out_tr})
    with rasterio.open(out_clipped, 'w', **out_meta) as dst:
        dst.write(out_img)
    print(f"Stitched and clipped {out_clipped}")

## 4. MERIT Hydro DEM (~90 m)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def get_image_from_config(cfg):
    """
    cfg: ['Image'|'ImageCollection', asset_id, band, name]
    Returns an ee.Image with the selected band.
    """
    kind, asset_id, band, *_ = cfg
    if kind == 'Image':
        return ee.Image(asset_id).select(band)
    elif kind == 'ImageCollection':
        return ee.ImageCollection(asset_id).select(band).mosaic()
    else:
        raise ValueError(f"Unknown kind: {kind}")

def build_box_from_point(lon, lat, buffer_m=40_000):
    """Buffered box around a point (like your notebook)."""
    pt = ee.Geometry.Point(lon, lat)
    return pt.buffer(buffer_m).bounds()

# ------------------------------------------------------------------
# Main downloader using xarray (fallback to geemap.ee_export_image)
# ------------------------------------------------------------------
def download_dem(dem_config, region_geom, out_path, scale=30, scale_factor=1):
    """
    dem_config: ['Image'|'ImageCollection', asset_id, band, name]
    region_geom: ee.Geometry
    out_path: output GeoTIFF path
    scale: pixel size for fallback export
    scale_factor: multiply the DEM values by this factor before export
                  (use 1000 for MERIT/DEM if your values are in km)
    """
    img = get_image_from_config(dem_config)

    # --- apply unit scaling here (affects BOTH xarray and geemap paths) ---
    if scale_factor != 1:
        img = img.multiply(scale_factor).toFloat().rename(f"{dem_config[2]}_scaled")

    img = img.clip(region_geom)    # clip to basin geometry

    try:
        ic = ee.ImageCollection(img)
        ds = xr.open_dataset(
            ic,
            engine="ee",
            projection=img.projection(),
            geometry=region_geom,
        )
        ds_t = ds.isel(time=0).drop_vars("time").transpose()
        ds_t.rio.set_spatial_dims("lon", "lat", inplace=True)
        ds_t.rio.to_raster(out_path)
        print(f"DEM saved (xarray path): {out_path}")

    except Exception as e:
        print(f"xarray path failed ({e}). Falling back to geemap.ee_export_image...")
        geemap.ee_export_image(
            img, filename=out_path, scale=scale, region=region_geom, file_per_band=False
        )
        print(f"DEM saved (geemap path): {out_path}")

# ------------------------------------------------------------------

dem_config = ['Image', 'MERIT/DEM/v1_0_3', 'dem', 'MERIT 30m']
dem_file = '/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/predictors/MERIT_DEM/MERIT30_dem.tif'
download_dem(dem_config, region_catch, dem_file, scale=30, scale_factor=1000)  # scale_factor=1000 for MERIT DEM in m


## 5. ERA5-Land daily temperature

def download_era5l_t2(catchments,
                      start="2000-01-01",
                      end="2024-12-31",
                      units="C",            # "C" or "K"
                      fmt="netcdf",         # "netcdf" or "zarr"
                      path="ERA5L_t2m.nc",
                      time_chunks=365,
                      buffer_m=10000,       # buffer in meters
                      clip_to_buffer=True,
                      use_bbox=True):

    # --- Build buffered bbox geometry ---
    base_geom = catchments.geometry().bounds() if use_bbox else catchments.geometry()
    buf_geom = base_geom.buffer(buffer_m) if buffer_m and buffer_m > 0 else base_geom
    open_geom = buf_geom.bounds() if use_bbox else buf_geom

    # --- ERA5-Land collection ---
    col0 = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
              .select("temperature_2m")
              .filterDate(ee.Date(start), ee.Date(end)))

    def prep(im):
        im2 = im.subtract(273.15) if units.upper() == "C" else im
        im2 = im2.rename("t2m")
        if clip_to_buffer:
            im2 = im2.clip(buf_geom)
        return im2.copyProperties(im, ["system:time_start", "system:time_end"])

    col = col0.map(prep)
    proj = ee.Image(col.first()).projection()

    # --- Open as xarray Dataset ---
    ds = xr.open_dataset(col, engine="ee", projection=proj, geometry=open_geom)

    # --- Rename dims to (time, lat, lon) ---
    ren = {}
    if "y" in ds.dims: ren["y"] = "lat"
    if "x" in ds.dims: ren["x"] = "lon"
    if ren: ds = ds.rename(ren)
    ds = ds.transpose("time", "lat", "lon")

    # --- Safe rechunk ---
    T, Y, L = ds.sizes["time"], ds.sizes["lat"], ds.sizes["lon"]
    ds = ds.chunk({"time": min(time_chunks, T),
                   "lat": min(256, Y),
                   "lon": min(256, L)})

    # --- Set attributes ---
    var = "t2m"
    ds[var] = ds[var].astype("float32")
    ds[var].attrs.update({
        "long_name": "2m air temperature (daily aggregated)",
        "units": "degC" if units.upper() == "C" else "K",
        "source": "ECMWF ERA5-Land Daily Aggregated (GEE: ECMWF/ERA5_LAND/DAILY_AGGR)"
    })

    # --- Export with progress bar ---
    with ProgressBar():
        if fmt.lower() == "netcdf":
            enc = {var: {"zlib": True, "complevel": 4, "shuffle": True,
                         "dtype": "float32",
                         "chunksizes": (min(time_chunks, T), min(256, Y), min(256, L))}}
            ds.to_netcdf(path, engine="netcdf4", encoding=enc)
        else:
            try:
                import numcodecs
                compressor = numcodecs.Blosc(cname="zstd", clevel=4,
                                             shuffle=numcodecs.Blosc.SHUFFLE)
                enc = {var: {"compressor": compressor, "dtype": "float32",
                             "chunks": (min(time_chunks, T), min(256, Y), min(256, L))}}
            except Exception:
                enc = {var: {"dtype": "float32",
                             "chunks": (min(time_chunks, T), min(256, Y), min(256, L))}}
            ds.to_zarr(path, mode="w", consolidated=True, encoding=enc)

    return path

era5_nc = download_era5l_t2(
    catchments,
    start="1999-01-01",
    end="2018-12-31",
    units="C",
    fmt="netcdf",
    path="ERA5L_t2m_1999_2018_bbox.nc",
    time_chunks=365,
    buffer_m=15000,        # e.g., 15 km buffer
    clip_to_buffer=True,   # clip to buffered bbox so edges are included
    use_bbox=True          # keep rectangular window for stability
)

## IDE execution

# download_glc()
# # Stitch annual GLC
# stitch_and_clip('GLC_annual_*_tile_*.zip', 'GLC_annual_mosaic.tif', 'GLC_annual_clipped.tif', gdf)
# # Stitch 5-year GLC
# stitch_and_clip('GLC_5yr_*_tile_*.zip', 'GLC_5yr_mosaic.tif', 'GLC_5yr_clipped.tif', gdf)

##
if __name__ == '__main__':
    download_modis_et()
    download_dem(dem_config, region_catch, dem_file, scale=30,
                 scale_factor=1000)
    era5_nc = download_era5l_t2(
        catchments,
        start="1999-01-01",
        end="2018-12-31",
        units="C",
        fmt="netcdf",
        path="ERA5L_t2m_1999_2018.nc",
        time_chunks=365
    )
    download_soil_hsg()
    download_glc()
    # Stitch annual GLC
    stitch_and_clip('GLC_annual_*_tile_*/*.tif', 'GLC_annual_mosaic.tif', 'GLC_annual_clipped.tif', gdf)
    # Stitch 5-year GLC
    stitch_and_clip('GLC_5yr_*_tile_*/*.tif', 'GLC_5yr_mosaic.tif', 'GLC_5yr_clipped.tif', gdf)

