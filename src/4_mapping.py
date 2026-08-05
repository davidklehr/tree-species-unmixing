# start pytohn code
import ast
import argparse
from datetime import datetime
import rasterio
import numpy as np
import os
import tensorflow as tf
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument("--dc_folder", help="path to the spline data-cube",
                     default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/ThermSpline_DC" )
parser.add_argument("--Gap_fraction_folder", help="path to the data of the canopy Gap",
                     default= "/data/ahsoka/eocp/wengler/ground_fractions/out" )
parser.add_argument("--working_directory", help="path to the pure data numpy array", 
                     default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix_CanopyGap")
parser.add_argument("--tree_class_list", help="labels of the tree species/classes in the correct order", 
                    #default = '[1,2,3,4,5,6,7,8,9,10,11,12,13,14]')
                    default = '[1,2,3,4,5,6,7,8,9,10,11,12,13]')
parser.add_argument("--tree_labels", help="labels of the tree species/classes in the correct order", 
                    default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Ahorn','Birke','Erle','OtherDT', 'Gap Fraction', 'Shadow']")
parser.add_argument("--num_models", help="number of models you want to create", default= '20')
parser.add_argument("--year", help="number of models you want to create", default= '2018')
parser.add_argument("--tile", help="The tile to be predicted", default= 'X0055_Y0053')
parser.add_argument("--forest_mask", help="path to the forest mask raster", 
                    #default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/cube/") # holzbodenkarte_2018
                    default= "/data/ahsoka/eocp/forestpulse/01_data/01_raw_data/germany/") # Germany
parser.add_argument("--local", help="check if model is calculated for a local tileset", 
                    default= 'FALSE')
args = parser.parse_args()
  
def get_stack(tile, year):
    file_path = os.path.join(args.dc_folder, tile, f'ThermSpline_coefs_{args.year}.tif')
    with rasterio.Env(GDAL_NUM_THREADS="1"):
        with rasterio.open(file_path) as src:
            stack = src.read()
            stack = np.moveaxis(stack, 0, -1)
    return stack

def predict(tile, year, no_of_tile, length):
    
    def norm(a):
        a_out = a/10000.
        return a_out
    
    def predict_model_on_tile(model, x_flat, static_in, forest_mask_flat, H, W):
        # Prüfen, ob gültige Pixel vorhanden sind
        if x_flat.shape[0] == 0:
            # Keine gültigen Daten, also leere Vorhersage zurückgeben
            return np.zeros((H, W, model.output_shape[-1]), dtype=np.float32)
        # Prediction
        preds = model([x_flat, static_in], training=False).numpy()
        y_pred = np.zeros((H * W, preds.shape[-1]), dtype=np.float32)
        y_pred[forest_mask_flat == 1] = preds
        # Rückkonvertieren ins Bildformat
        y_pred = y_pred.reshape((H, W, -1))  # Shape: (H, W, N_CLASSES)
        return y_pred
    
    # =============================================
    # define input (if present) and output
    # =============================================
    input_raster = os.path.join(args.dc_folder, tile, f'ThermSpline_coefs_{args.year}.tif')
    

    if not os.path.isfile(input_raster):
        print(os.path.join(args.dc_folder, tile))
        print(no_of_tile, 'Status: ' + tile + f' [{no_of_tile}/{length}]' + ' Not tile - skipped!')
        print('Not tile (ThermalSpline), skipping!')
        return
    
    # start processing here
    print('Status : ' + tile + f' [{no_of_tile}/{length}]' + ' prediction started...')
    start=datetime.now()
    x_in = get_stack(tile, year)

    # forest mask
    mask_path = os.path.join(args.forest_mask, tile, f'germany.tif')
    if not os.path.isfile(mask_path):
        print(no_of_tile, 'Status: ' + tile + f' [{no_of_tile}/{length}]' + ' No forest mask - skipped!')
        print('No forest mask, skipping!')
        return
    with rasterio.open(mask_path) as src:
        meta = src.meta.copy()
        forest_mask = src.read(1)
    forest_mask_flat = forest_mask.flatten()

    meta.update(
        count=len(ast.literal_eval(args.tree_labels)),  # Anzahl der Layer entspricht der Anzahl der TIFFs
        dtype='uint8',          # 8-Bit Integer
        compress='ZSTD'          # LZW-Komprimierung
    )

    # apply forest_mask
    H, W, bands = x_in.shape
    x_flat = x_in.reshape((-1, bands))
    x_valid = x_flat[forest_mask_flat == 1]

    ground_raster_path = os.path.join(args.Gap_fraction_folder, args.year, tile ,f'gapfraction_{args.year}.tif' )
    if not os.path.isfile(ground_raster_path):
        print(no_of_tile, 'Status: ' + tile + f' [{no_of_tile}/{length}]' + ' Not tile - skipped!')
        print('Not tile (ground fraction), skipping!')
        return
    with rasterio.open(ground_raster_path) as src:
        ground_data = src.read(1)
    ground_data = ground_data / 100
    static_flat = ground_data.reshape((-1, 1))
    static_valid = static_flat[forest_mask_flat == 1]
    x_valid = norm(x_valid.astype(np.float32))

    # =============================================
    # load model list
    # =============================================
    from keras.saving import register_keras_serializable
    @register_keras_serializable(package="Custom")
    class SumToOneLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            return inputs / tf.reduce_sum(inputs, axis=-1, keepdims=True)

    model_list = []
    for i in range(int(args.num_models)):
        if args.local == 'TRUE':
            model_path = os.path.join(args.working_directory, args.year,'3_trained_model_tile', tile , 'version' +str(i+1), 'saved_model'+ str(i+1)+ '.keras')
        else:
            model_path = os.path.join(args.working_directory, args.year ,'3_trained_model_glob', 'version' +str(i+1), 'saved_model'+ str(i+1)+ '.keras')
        model = tf.keras.models.load_model(model_path)
        model_list.append(model)

    y_out = np.zeros([H, W, len(ast.literal_eval(args.tree_class_list))]) 
    name_list = ast.literal_eval(args.tree_labels)
    #print('Data normed',datetime.now()-start) # at about 3:00

    # =============================================
    #          multi model prediction
    # =============================================
    print('Status: ' + tile + f' [{no_of_tile}/{length}]' + ' multi model prediction...')
    list_predictions =[]
    for model in model_list:
        y_out = predict_model_on_tile(model, x_valid, static_valid, forest_mask_flat, H, W)
        y_out = np.clip(y_out * 100, a_min = 0, a_max = None)
        list_predictions.append(np.copy(y_out))
    stacked_arrays = np.stack(list_predictions, axis=-1)

    # =============================================
    #          median fraction and deviation
    # =============================================
    average_array = np.mean(stacked_arrays, axis=-1)
    median_array = np.median(stacked_arrays, axis=-1)
    #deviation = np.mean(np.absolute(stacked_arrays - average_array[..., np.newaxis]), axis=-1)
    
    # classification of dominant species
    y_out_clf = np.argmax(median_array, axis= -1)
    y_out_clf += 1
    y_out_clf[forest_mask != 1] = 255
    y_out_clf = y_out_clf.astype(np.int8)

    median_array[forest_mask != 1] = 255
    median_array = median_array.astype(np.int8)

    # =============================================
    #              writing outputs 
    # =============================================       
    if args.local == 'TRUE':
        out_dir = os.path.join(args.working_directory, args.year, '4_prediction', tile)
    else:
        out_dir = os.path.join(args.working_directory, args.year, '4_prediction_glob', tile)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #---------- 1. median fraction ------------
    with rasterio.open(os.path.join(out_dir, f'{args.year}_fraction.tif'), 
                       'w', **meta) as dst:
        dst.descriptions = name_list
        for i in range(median_array.shape[-1]):
            dst.write(median_array[..., i].astype(np.uint8), i+1)
            dst.set_band_description(i+1, name_list[i])

    # --------- 2. classification -------------
    meta.update(
        count=1,  
        dtype='uint8',          # 8-Bit Integer
        compress='ZSTD'          # LZW-Komprimierung
    )
    with rasterio.open(os.path.join(out_dir, f'{args.year}_dominant.tif'), 
                       'w', **meta) as dst:
        dst.write(y_out_clf, 1)
        dst.set_band_description(1, "dominant species")

    # -------- 3. deviation --------------
    # ... pending ...
    print('Status: ' + tile + f' [{no_of_tile}/{length}]' + ' prediction finished... | Duration: ' + str(datetime.now()-start))

if __name__ == '__main__':
    os.makedirs(os.path.join(args.working_directory, args.year, '4_prediction_glob'), exist_ok=True)
    os.makedirs(os.path.join(args.working_directory, '4_prediction'), exist_ok=True)
    # ----------- Predict just one tile  --------
    #predict(args.tile, args.year , 1, 1)
    # call via:
    # parallel -j 5 python 4_mapping.py --tile {} :::: /data/ahsoka/eocp/forestpulse/02_scripts/DWD/RLP_tilelist.txt
    if not os.path.isfile(os.path.join(args.working_directory, args.year, '4_prediction_glob' ,args.tile, f'{args.year}_fraction.tif')):
        predict(args.tile.strip(), args.year , 1, 1)
    #--------------------------------------------
    
