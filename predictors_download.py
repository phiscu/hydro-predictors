import os
import pyproj
import ee
import geopandas as gpd
import geemap
import requests
import glob
import zipfile
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
# To avoid PROJ error:
os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()

# Initialize Earth Engine
ee.Initialize(project="matilda-edu")

# 1. Load local catchments from a GeoPackage and convert to EE FeatureCollection
# Replace 'catchments.gpkg' and 'your_layer_name' with your file and layer
gdf = gpd.read_file('/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/geodata/issykul_vectors.gpkg', layer='catchment_new')
catchments = geemap.geopandas_to_ee(gdf)

# Define regions
def get_region_catch():
    return catchments.geometry().getInfo()

def get_region_bbox():
    return catchments.geometry().bounds().getInfo()

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


## IDE execution

# download_glc()
# # Stitch annual GLC
# stitch_and_clip('GLC_annual_*_tile_*.zip', 'GLC_annual_mosaic.tif', 'GLC_annual_clipped.tif', gdf)
# # Stitch 5-year GLC
# stitch_and_clip('GLC_5yr_*_tile_*.zip', 'GLC_5yr_mosaic.tif', 'GLC_5yr_clipped.tif', gdf)


##
if __name__ == '__main__':
    download_modis_et()
    download_soil_hsg()
    download_glc()
    # Stitch annual GLC
    stitch_and_clip('GLC_annual_*_tile_*/*.tif', 'GLC_annual_mosaic.tif', 'GLC_annual_clipped.tif', gdf)
    # Stitch 5-year GLC
    stitch_and_clip('GLC_5yr_*_tile_*/*.tif', 'GLC_5yr_mosaic.tif', 'GLC_5yr_clipped.tif', gdf)

