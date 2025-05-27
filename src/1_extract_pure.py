#! /home/ahsoka/klehr/anaconda3/envs/Synth_Mix/bin python3
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dc_folder", help="path to the spline data-cube", default= "/data/ahsoka/eocp/forestpulse/INTERNAL/spline/5day_interval/thermal" )
parser.add_argument("--training_points", help="path to the file of the training points geopackage", default= "/data/ahsoka/eocp/forestpulse/INTERNAL/BWI4/all_trainings_points.gpkg")
parser.add_argument("--year", help="path to the file of the training points geopackage", default= 2021)
parser.add_argument("--working_directory", help="path to the file of the training points geopackage", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix/2021_ThermalTime")
args = parser.parse_args()


bands = ['BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'BNIR', 'NIR', 'SW1', 'SW2']
vrt_paths = {
    band: os.path.join(args.dc_folder, f"mosaic/stack_{band}.vrt")
    for band in bands
}
    
# 1. load point data
print(args.training_points)
gdf = gpd.read_file(args.training_points)
gdf = gdf.to_crs("EPSG:3035")

arr_x = []
arr_y = []

# 2. opan all vrts
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
    # 3. save (iteration-wise)
    arr_x_out = np.array(arr_x)
    arr_y_out = np.array(arr_y)
    np.save(os.path.join(args.working_directory, '1_pure' , f"x_{str(args.year)}.npy"), arr_x_out)
    np.save(os.path.join(args.working_directory, '1_pure' , f"y_{str(args.year)}.npy"), arr_y_out)

# close Raster-data
for ds in datasets.values():
    ds.close()
