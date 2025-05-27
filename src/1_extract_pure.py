import geopandas as gpd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
import os
from params_2021_thermal_time import params
import argparse


bands = ['BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'BNIR', 'NIR', 'SW1', 'SW2']
vrt_paths = {
    band: f"./INTERNAL/spline/5day_interval/thermal/mosaic/stack_{band}.vrt"
    for band in bands
}
    
# 1. Lade Punkte
gdf = gpd.read_file("./INTERNAL/BWI4/all_trainings_points.gpkg")
gdf = gdf.to_crs("EPSG:3035")

arr_x = []
arr_y = []

# 2. Öffne alle vrts einmal
datasets = {band: rasterio.open(path) for band, path in vrt_paths.items()}
n_layers = datasets[bands[0]].count

for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Processing samples"):
    # iterate over every data point:
    coords = row.geometry
    spec = row.spec
    point_data = []
    for band in bands:
        # get the raster band vrt
        dataset = datasets[band]
        try:
            values = []
            row_idx, col_idx = dataset.index(coords.x, coords.y)
            # get data from bands 1 to 28
            for i in range(1, n_layers + 1):  # bands 1 to 28
                v = dataset.read(i, window=Window(col_idx, row_idx, 1, 1))[0, 0]
                values.append(v)
            values = np.array(values) # dimension [,28]
        except Exception:
            values = np.full(n_layers, np.nan)
        point_data.append(values) # append all bands
    sample_array = np.stack(point_data, axis=-1) # becomes an array [28,10]
    arr_x.append(sample_array)
    arr_y.append(spec)
    # 3. Speichern
    arr_x_out = np.array(arr_x)
    arr_y_out = np.array(arr_y)
    np.save(os.path.join('./02_scripts/Synth_Mix/visualization/thermal', "x_arr.npy"), arr_x_out)
    np.save(os.path.join('./02_scripts/Synth_Mix/visualization/thermal', "y_arr.npy"), arr_y_out)

# Schließe Raster-Dateien
for ds in datasets.values():
    ds.close()

# 3. Speichern
arr_x = np.array(arr_x)
arr_y = np.array(arr_y)
print(arr_x.shape)
print(arr_x[173,:,0])
print(arr_x[173,:,2])
print(arr_x[173,:,7])
print(arr_y.shape)
np.save(os.path.join('./02_scripts/Synth_Mix/visualization/thermal', "x_arr.npy"), arr_x)
np.save(os.path.join('./02_scripts/Synth_Mix/visualization/thermal', "y_arr.npy"), arr_y)
