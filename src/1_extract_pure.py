#! /home/ahsoka/klehr/anaconda3/envs/Synth_Mix/bin python3
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
import os
import argparse
from joblib import Parallel, delayed
from shapely.geometry import box

parser = argparse.ArgumentParser()
parser.add_argument("--dc_folder", help="path to the spline coefficents data-cube", 
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/ThermSpline_DC" )
parser.add_argument("--training_points", help="path to the file of the training points geopackage", 
                    #default= "/data/ahsoka/eocp/forestpulse/INTERNAL/BWI4/2nd_sampling/3_GIS_selection/4th_sampling_merge.gpkg") # for 2021
                    default= "/data/ahsoka/eocp/forestpulse/INTERNAL/BWI4/2nd_sampling/3_GIS_selection/4th_sampling_merge_no_Poplar.gpkg")
parser.add_argument("--year", help="path to the file of the training points geopackage", default= '2025')
parser.add_argument("--working_directory", help="path to the file of the training points geopackage", 
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix_CanopyGap/2025")
parser.add_argument("--tile", help="The tile to be predicted", 
                    default= 'X0055_Y0053')
args = parser.parse_args()

def extract_points(tile):
    dc_path = os.path.join(args.dc_folder, tile)
    file_path = os.path.join(dc_path, f'ThermSpline_coefs_{args.year}.tif')
    if not os.path.exists(file_path):
        return

    with rasterio.open(file_path) as src:
        band_1 = src.read(1)  # get raster information (position, height, width etc.)
            
        # 1. load point data
        gdf = gpd.read_file(args.training_points)
        gdf = gdf.to_crs("EPSG:3035")

        #for fid, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Processing samples"):
        for fid, row in gdf.iterrows():
            # iterate over every data point:
            coords = row.geometry
            spec = row.Spec
            point_data = []
            try:
                # extract data, read the raster in windows here, for performance
                values = []
                row_idx, col_idx = src.index(coords.x, coords.y)
                if (row_idx >= 0) and (col_idx >= 0) and (row_idx < band_1.shape[0]) and (col_idx < band_1.shape[1]):
                    for band in range(220):
                        v = src.read(band+1, window=Window(col_idx, row_idx, 1, 1))[0, 0]
                        values.append(v)
                    values = np.array(values) # dimension [,220]
                    print(row.PlotID)  
                    print(fid)
                    print(values[0:10]) 
            except Exception:
                values = np.full(220, np.nan)

            point_data.append(values) # append all values
            sample_array = np.stack(point_data, axis=-1) # becomes an array 
            # ------- store data --------
            if not np.isnan(sample_array).all():
                print(fid)
                print(os.path.join(args.working_directory, args.year, '1_pure', f'samples_x{str(args.year)}',f'x_{str(fid).zfill(4)}.csv'))
                #----- maybe add reshape later here -------
                np.savetxt(os.path.join(args.working_directory, args.year, '1_pure', f'samples_x{str(args.year)}',f'x_{str(fid).zfill(4)}.csv'), sample_array, delimiter=",", fmt="%d")
                np.savetxt(os.path.join(args.working_directory, args.year, '1_pure', f'samples_y{str(args.year)}',f'y_{str(fid).zfill(4)}.csv'), [np.array(spec)], fmt="%d")

def to_point(geom):
    if geom.geom_type == "MultiPoint":
        return list(geom.geoms)[0]
    return geom

def sample_points(tile):
    dc_path = os.path.join(args.dc_folder, tile)
    file_path = os.path.join(dc_path, f'ThermSpline_coefs_{args.year}.tif')
    if not os.path.exists(file_path):
        return
    
    with rasterio.open(file_path) as src:
        bounds = src.bounds 
        raster_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            
        # 1. load point data
        gdf = gpd.read_file(args.training_points)
        gdf = gdf.to_crs("EPSG:3035")

        # 2. clip the point data to the raster bounds
        gdf_clipped = gdf[gdf.intersects(raster_geom)].copy()
        gdf_clipped["geometry"] = gdf_clipped.geometry.apply(to_point)

        # 3. extract raster values at the point locations
        for fid, row in gdf_clipped.iterrows():
            coords = (row.geometry.x, row.geometry.y)
            tree_species = row["Spec"]
            sample = list(src.sample([coords]))[0] # extract the sample values for the point
            # 4. save the extracted samples
            np.savetxt(os.path.join(args.working_directory, args.year, '1_pure', f'samples_x{str(args.year)}',
                                    f'x_{str(fid).zfill(4)}.csv'), sample, delimiter=",", fmt="%d")
            np.savetxt(os.path.join(args.working_directory, args.year, '1_pure', f'samples_y{str(args.year)}',
                                    f'y_{str(fid).zfill(4)}.csv'), [np.array(tree_species)], fmt="%d")

    
if __name__ == '__main__':
    os.makedirs(os.path.join(args.working_directory, args.year, '1_pure', f'samples_x{str(args.year)}'), exist_ok=True))
    os.makedirs(os.path.join(args.working_directory, args.year, '1_pure', f'samples_y{str(args.year)}'), exist_ok=True))

    #--------------------------------------
    # sinle tile processing for testing
    # parallel -j 5 python 1_extract_pure.py --tile {} :::: /data/ahsoka/eocp/forestpulse/02_scripts/DWD/RLP_tilelist.txt
    sample_points(args.tile.strip())
    #--------------------------------------

    #--------------------------------------
    # parallelized processing
    #tiles = [tile for tile in os.listdir(args.dc_folder) if tile.startswith("X")]
    #Parallel(n_jobs=15)(delayed(sample_points)(tile) for tile in tqdm(tiles, desc="Processing tiles"))
    #--------------------------------------
