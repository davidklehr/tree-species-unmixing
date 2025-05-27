# for paralellization 
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1
import os

def set_threads(
    num_threads,
    set_blas_threads=True,
    set_numexpr_threads=True,
    set_openmp_threads=False
):
    num_threads = str(num_threads)
    if not num_threads.isdigit():
        raise ValueError("Number of threads must be an integer.")
    if set_blas_threads:
        os.environ["OPENBLAS_NUM_THREADS"] = num_threads
        os.environ["MKL_NUM_THREADS"] = num_threads
        os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
    if set_numexpr_threads:
        os.environ["NUMEXPR_NUM_THREADS"] = num_threads
    if set_openmp_threads:
        os.environ["OMP_NUM_THREADS"] = num_threads

set_threads(1)

# start pytohn code
import tensorflow as tf
import numpy as np
from osgeo import gdal
import rasterio
import ast
import argparse
from datetime import datetime
from keras.saving import register_keras_serializable
from joblib import Parallel, parallel_backend, delayed

parser = argparse.ArgumentParser()
parser.add_argument("--dc_folder", help="path to the spline data-cube", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/ThermalTime_Spline" )
parser.add_argument("--working_directory", help="path to the pure data numpy array", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix/2021_ThermalTime")
parser.add_argument("--tree_class_list", help="labels of the tree species/classes in the correct order", default = '[1,2,3,4,5,6,7,8,9,10,11,12,13,14]')
parser.add_argument("--tree_labels", help="labels of the tree species/classes in the correct order", default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Ahorn','Birke','Erle','Pappel','Weide', 'Ground', 'Shadow']")
parser.add_argument("--num_models", help="number of models you want to create", default= 5)
parser.add_argument("--year", help="number of models you want to create", default= '2021')
args = parser.parse_args()

@register_keras_serializable(package="Custom")
class SumToOneLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs / tf.reduce_sum(inputs, axis=-1, keepdims=True)
    
def get_stack(tile, year):
    def get_band(band_name):
        band_path = os.path.join(args.dc_folder, tile, 'stack_{band_name}.vrt'.format(band_name=band_name))
        with rasterio.open(band_path) as src:
            band = src.read()
        band = np.moveaxis(band, 0, -1)
        return band
    band_list = ['BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'BNIR', 'NIR', 'SW1', 'SW2']
    stack = np.array([get_band(b) for b in band_list])
    stack = np.moveaxis(stack, 0, -1)
    return stack

def predict(tile, year, model_list, no_of_tile, length):

    #@register_keras_serializable(package="Custom")
    #class SumToOneLayer(tf.keras.layers.Layer):
    #    def call(self, inputs):
    #        return inputs / tf.reduce_sum(inputs, axis=-1, keepdims=True)
    #model_list = []
    #for i in [0,1,2]:
    #    model_path = os.path.join(args.working_directory, '3_trained_model_test' ,'version' +str(i+1),'saved_model'+ str(i+1)+ '.keras') 
    #    model = tf.keras.models.load_model(model_path, custom_objects={'SumToOneLayer': SumToOneLayer} )
    #    model_list.append(model)
        
    def pred(model, x):
        y_pred = model(x, training=False)
        return y_pred.numpy()
    def norm(a):
        a_out = a/10000.
        return a_out
    def toRasterFraction(arr_in, name_list):
        path = os.path.join(args.dc_folder, tile, 'stack_BLU.vrt')
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        [cols, rows] = arr.shape
        bands = arr_in.shape[-1]
        driver = gdal.GetDriverByName("GTiff")
        path_out = path = os.path.join(args.working_directory, '4_prediction_test', tile, 'fraction_' + year + '.tif')
        print(path_out)
        outdata = driver.Create(path_out, rows, cols, bands, gdal.GDT_Byte)
        #outdata = driver.Create(path_out, rows, cols, bands, gdal.GDT_Float32)
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input
        for i in range(bands):
            outdata.GetRasterBand(i + 1).WriteArray(arr_in[..., i])
            outdata.GetRasterBand(i + 1).SetNoDataValue(255)
            outdata.GetRasterBand(i + 1).SetDescription(name_list[i])
        outdata.FlushCache() ##saves to disk!!
        outdata = None
        band=None
        ds=None

    def toRasterDeviation(arr_in, name_list):
        path = os.path.join(args.dc_folder, tile, 'stack_BLU.vrt')
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        [cols, rows] = arr.shape
        bands = arr_in.shape[-1]
        driver = gdal.GetDriverByName("GTiff")
        path_out = path = os.path.join(args.working_directory, '4_prediction_test', tile, 'deviation_' + year + '.tif')
        print(path_out)
        outdata = driver.Create(path_out, rows, cols, bands, gdal.GDT_Float32)
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input
        for i in range(bands):
            outdata.GetRasterBand(i + 1).WriteArray(arr_in[..., i])
            outdata.GetRasterBand(i + 1).SetNoDataValue(255)
            outdata.GetRasterBand(i + 1).SetDescription(name_list[i])
        outdata.FlushCache() ##saves to disk!!
        outdata = None
        band=None
        ds=None

    def toRasterClassification(arr_in):
        path = os.path.join(args.dc_folder, tile, 'stack_BLU.vrt')
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        [cols, rows] = arr.shape
        driver = gdal.GetDriverByName("GTiff")
        path_out = os.path.join(args.working_directory, '4_prediction_test', tile, 'classification_' + year + '.tif')
        outdata = driver.Create(path_out, rows, cols, 1, gdal.GDT_Byte)
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(arr_in)
        outdata.GetRasterBand(1).SetNoDataValue(255)
        outdata.FlushCache() ##saves to disk!!
        outdata = None
        band=None
        ds=None

    # =============================================
    # define input (if present) and output
    # =============================================
    blue_band = os.path.join(args.dc_folder, tile, 'stack_BLU.vrt')
    if not os.path.isfile(blue_band):
        print('Not tile, skipping!')
        return
    out_dir = os.path.join(args.working_directory, '4_prediction_test', tile)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # start processing here
    print('Predicting', tile, year, '...', str(no_of_tile+1) , '/{number} ----------'.format(number=length-1),sep = ' ')
    start=datetime.now()

    x_in = get_stack(tile, year)
    nodata_mask = x_in[:, :, 0, 0] == -9999
    x_in = norm(x_in.astype(np.float32))
   
    y_out = np.zeros([x_in.shape[0], x_in.shape[1], len(ast.literal_eval(args.tree_class_list))]) 
    name_list = ast.literal_eval(args.tree_labels)

    # =============================================
    #          multi model prediction
    # =============================================
    list_predictions =[]
    model_num = 0 
    for model in model_list:
        model_num =  model_num + 1
        for i in range(y_out.shape[1]):
            x_batch = x_in[i, ...]
            y_out[i, ...] = pred(model, x_batch)
        y_out = y_out * 100
        y_out[y_out < 0.] = 0.
        y_out[y_out > 100.] = 100.
        list_predictions.append(np.copy(y_out))
        single = y_out
        single[nodata_mask] = 255
    stacked_arrays = np.stack(list_predictions, axis=-1)
    # median fraction and deviation
    #average_array = np.mean(stacked_arrays, axis=-1)
    median_array = np.median(stacked_arrays, axis=-1)
    #deviation = np.mean(np.absolute(stacked_arrays - average_array[..., np.newaxis]), axis=-1)

    # classification of dominant species
    y_out_clf = np.argmax(median_array, axis= -1)
    y_out_clf += 1
    y_out_clf[nodata_mask] = 255
    y_out_clf = y_out_clf.astype(np.int8)

    # ===============
    # writing outputs 
    # ===============
    # median
    median_array[nodata_mask] = 255
    median_array = median_array.astype(np.int8)   
    toRasterFraction(median_array, name_list)
    # deviation
    #deviation[nodata_mask] = 255
    #deviation = deviation.astype(np.int8)
    #toRasterDeviation(deviation, name_list)
    
    toRasterClassification(y_out_clf)
    print('-------- Predicting ' + tile + ' done successfully  | ' + str(no_of_tile) + '/{number} | Duration: '.format(number=length-1) + str(datetime.now()-start) + ' ----------')

if __name__ == '__main__':
    
    model_list = []
    #for i in range(args.num_models):
    for i in [0,1,2]:
        model_path = os.path.join(args.working_directory, '3_trained_model_test' ,'version' +str(i+1),'saved_model'+ str(i+1)+ '.keras') 
        model = tf.keras.models.load_model(model_path, custom_objects={'SumToOneLayer': SumToOneLayer} )
        model_list.append(model)

    list_tiles = os.listdir(args.dc_folder)
    list_tiles = ['X0055_Y0053','X0055_Y0054']
    year = int(args.year)
    with parallel_backend('threading'):
        Parallel(n_jobs=2)(delayed(predict)(tile, '2021', model_list, list_tiles.index(tile), len(list_tiles)) for tile in list_tiles)