#! /home/ahsoka/klehr/anaconda3/envs/Synth_Mix/bin python3
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dc_folder", help="path to the spline data-cube", default= "/data/ahsoka/eocp/forestpulse/INTERNAL/spline/thermal_time_training_points" )
parser.add_argument("--training_points", help="path to the file of the training points geopackage", default= "/data/ahsoka/eocp/forestpulse/INTERNAL/BWI4/all_trainings_points.gpkg")
parser.add_argument("--year", help="path to the file of the training points geopackage", default= 2021)
parser.add_argument("--working_directory", help="path to the file of the training points geopackage", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix/2021_ThermalTime")
args = parser.parse_args()

def extract_points(tile):
    dc_path = os.path.join(args.dc_folder, tile)
    bands = ['BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'BNIR', 'NIR', 'SW1', 'SW2']
    data_by_band = {}
    for band in bands:
        band_paths =[]
        for datei in os.listdir(dc_path):
            if ('_'+band+'_' in datei) and datei.endswith('.tif'):
                path = os.path.join(dc_path, datei)
                band_paths.append(path)   
        band_paths = sorted(band_paths)
        band_data = [rasterio.open(path) for path in band_paths]
        data_by_band[band] = band_data
        
    # 1. load point data
    gdf = gpd.read_file(args.training_points)
    gdf = gdf.to_crs("EPSG:3035")

    #for fid, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Processing samples"):
    for fid, row in gdf.iterrows():
        # iterate over every data point:
        coords = row.geometry
        spec = row.spec
        point_data = []
        for band in bands:
            # get the raster band
            all_dates = data_by_band[band]
            try:
                values = []
                row_idx, col_idx = all_dates[0].index(coords.x, coords.y)
                # get data from bands 1 to 28
                for data_at_date in all_dates:   # bands 1 to 28
                    v = data_at_date.read(1, window=Window(col_idx, row_idx, 1, 1))[0, 0]
                    values.append(v)
                values = np.array(values) # dimension [,28]
            except Exception:
                values = np.full(len(all_dates), np.nan)
            point_data.append(values) # append all bands
        sample_array = np.stack(point_data, axis=-1) # becomes an array [28,10]
        if not np.isnan(sample_array).all():
            np.savetxt(os.path.join(args.working_directory, '1_pure', f'samples_x{str(args.year)}',f'x_{str(fid).zfill(4)}.csv'), sample_array, delimiter=",", fmt="%d")
            np.savetxt(os.path.join(args.working_directory, '1_pure', f'samples_y{str(args.year)}',f'y_{str(fid).zfill(4)}.csv'), [np.array(spec)], fmt="%d")

    # close Raster-data
    for band in bands:
        all_dates = data_by_band[band]
        for data_at_date in all_dates: 
            data_at_date.close()

if __name__ == '__main__':
    if not os.path.exists(os.path.join(args.working_directory, '1_pure', f'samples_x{str(args.year)}')):
        os.makedirs(os.path.join(args.working_directory, '1_pure', f'samples_x{str(args.year)}'))
    if not os.path.exists(os.path.join(args.working_directory, '1_pure', f'samples_y{str(args.year)}')):
        os.makedirs(os.path.join(args.working_directory, '1_pure', f'samples_y{str(args.year)}'))
    extract_points(tile)
