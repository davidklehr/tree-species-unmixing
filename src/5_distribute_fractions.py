#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 2024
@author: klehr
"""
import os
from joblib import Parallel, delayed

from osgeo import gdal
import numpy as np
import os
import rasterio
from rasterio.mask import mask
from time import process_time
from tqdm import tqdm
import argparse
import ast
from scipy.ndimage import convolve

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the pure data numpy array", 
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix_CanopyGap")
parser.add_argument("--noisy_th", help="threshold for noisy prediction value", default= "20")
parser.add_argument("--forest_mask_folder", help="path to the forest mask", 
                    #default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/cube")
                    default= "/data/ahsoka/eocp/forestpulse/01_data/01_raw_data/germany/") # Germany
parser.add_argument("--forest_mask_name", help="name of the forest mask raster", 
                    #default= "holzbodenkarte_2018.tif")
                    default= "germany.tif")      
parser.add_argument("--use_disturbance_mask", help="should a disturbance/bb maks be used?", default= "T")
parser.add_argument("--disturbance_mask_folder", help="path to the forest mask", 
                    default= '/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/hungry-beetle/HungryBeetle_DEU')
parser.add_argument("--disturbance_mask_name", help="name of the forest mask raster", default= 'disturbance_year.tif')
parser.add_argument("--tree_class_list", help="labels of the tree species/classes in the correct order", 
                    #default = '[1,2,3,4,5,6,7,8,9,10,11,12,13,14]')
                    default = '[1,2,3,4,5,6,7,8,9,10,11,12,13]')
parser.add_argument("--tree_labels", help="labels of the tree species/classes in the correct order", 
                    default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Ahorn','Birke','Erle','OtherDT', 'Canopy Gap']")
parser.add_argument("--year", help="number of models you want to create", default= '2018')
parser.add_argument("--tile", help="The tile to be normalize", default= 'X0053_Y0053')
parser.add_argument("--local", help="check if model is calculated for a local tileset", default= 'FALSE')
args = parser.parse_args()


def normalize_bands(tile):
    if args.local == 'TRUE':
        input_path = os.path.join(args.working_directory, args.year, '4_prediction', tile,'{year}_fraction.tif'.format(year = int(args.year)))
        output_path = os.path.join(args.working_directory, args.year, '5_prediction_distributed', tile)
    else:
        input_path = os.path.join(args.working_directory, args.year, '4_prediction_glob', tile,'{year}_fraction.tif'.format(year = int(args.year)))
        output_path = os.path.join(args.working_directory, args.year, '5_prediction_distributed_glob', tile)

    os.makedirs(output_path, exist_ok=True)

    if not os.path.isfile(input_path):
        print(input_path)
        print(f'No prediction raster for tile {tile}, skipping normalization!')
        return    
    # ----------------- load forest mask -----------------
    mask_path = os.path.join(args.forest_mask_folder, tile ,args.forest_mask_name ) 
    if not os.path.isfile(mask_path):
        print(f'No forest mask for tile {tile}, skipping normalization!')
        return
    with rasterio.open(mask_path) as mask_src:
        forest_mask = mask_src.read(1)
        # 0 = no forest; 1 = forest
    
    # ----------------- load disturbance mask -----------------
    # bb_mask_path = os.path.join(args.disturbance_mask_folder, tile, 'disturbance' ,args.disturbance_mask_name ) 
    # if not os.path.isfile(bb_mask_path):
    #     print(f'No disturbance mask for tile {tile}, skipping normalization!')
    #     return
    # with rasterio.open(bb_mask_path) as mask_src:
    #     bb_mask = mask_src.read(1)
        
    # ----------------- load data and calculate sum -----------------
    with rasterio.open(input_path) as src:
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                 "compress":'ZSTD',
                 "dtype": rasterio.uint8,
                 "count": len(ast.literal_eval(args.tree_labels))
                 })
        band_values = []

        for band_num in range(1, len(ast.literal_eval(args.tree_class_list))-1): # no Gap Fraction and shadow
            band_array = src.read(band_num)
            band_array[(band_array > 0) & (band_array <= int(args.noisy_th) )] = 0
            band_array[band_array == 255] = 0
            band_values.append(band_array)
        
        # Calculate the sum of all band values for each pixel
        tree_species_stack = np.stack(band_values)
        tree_species_sum = np.sum(tree_species_stack, axis=0)

        shadow_array = src.read(len(ast.literal_eval(args.tree_class_list)))
        gap_array = src.read(len(ast.literal_eval(args.tree_class_list))-1)
        crown_array = 100 - gap_array

        # mask where shadow is > 0 and tree_speceiss_sum is = 0
        shadow_dist_mask = (shadow_array > 0) & (tree_species_sum == 0)
        # define kernel
        kernel = np.full((5, 5), 1/160, dtype=float)
        kernel[1:4, 1:4] = 0.9/8
        kernel[2, 2] = 0

        for band_num in range(1, len(ast.literal_eval(args.tree_class_list))-1): # no Gap Fraction and shadow
            s = convolve(tree_species_stack[band_num-1,:, :].astype(float), kernel, mode="nearest")
            tree_species_stack[band_num-1,shadow_dist_mask] = s[shadow_dist_mask]
        total_sum = np.sum(tree_species_stack, axis=0)

        # ----------------- normalize each band and store it in the output -----------------
        with rasterio.open(os.path.join(output_path , 'tree_fraction_and_gap.tif'), "w" , **out_meta) as dest:
            dest.descriptions = tuple( ast.literal_eval(args.tree_labels) )
            for band_num in range(tree_species_stack.shape[0]):
                with np.errstate(divide='ignore', invalid='ignore'):
                    #normalized_band = tree_species_stack[band_num, :,:] / total_sum
                    normalized_band = np.divide(tree_species_stack[band_num, :,:], total_sum)
                    rounded_band = np.round(np.multiply(normalized_band, crown_array)).astype(np.uint8)
                    
                rounded_band[total_sum == 0] = 0
                # clip to forest mask
                rounded_band[forest_mask==0] = 255
                rounded_band[forest_mask==255] = 255 # outside germany
                rounded_band = rounded_band.astype(np.uint8)
                # clip to disturbance mask
                #if (args.use_disturbance_mask == 'T'):
                #    rounded_band[(bb_mask > 1) & (bb_mask <= 2021)] = 255
                dest.write(rounded_band, band_num+1)
            # add gap fraction band
            gap_array[forest_mask == 0] = 255
            gap_array[forest_mask == 255] = 255
            #if (args.use_disturbance_mask == 'T'):
            #        gap_array[(bb_mask > 1) & (bb_mask <= 2021)] = 255
            dest.write(gap_array, src.count-1)

if __name__ == "__main__":
    #parallel --line-buffer -j 32 python -u 5_distribute_fractions.py --year 2018 --tile {} :::: /data/ahsoka/eocp/forestpulse/02_scripts/DWD/RLP_tilelist.txt
    normalize_bands(args.tile.strip())

    
