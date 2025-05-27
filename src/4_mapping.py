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
import logging
logging.disable(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from osgeo import gdal
import rasterio
from tqdm import tqdm
from params_avg import params
from joblib import Parallel, delayed

def get_stack(tile, year):
    def get_band(band_name):
        y1 = int(year)-3
        y2 = int(year)+1
        band_path = os.path.join(params['DATA_CUBE_DIR'], tile, '{y1}-{y2}_001-365_HL_TSA_SEN2L_{band_name}_RSP.tif'.format(y1=y1, y2=y2,band_name=band_name))
        with rasterio.open(band_path) as src:
            band = src.read()
        band = np.moveaxis(band, 0, -1)
        return band
    band_list = params['BAND_LIST']
    stack = np.array([get_band(b) for b in band_list])
    stack = np.moveaxis(stack, 0, -1)
    return stack

def predict(tile, year, model_list, no_of_tile):
    def pred(model, x):
        y_pred = model(x, training=False)
        return y_pred.numpy()
    def norm(a):
        a_out = a/10000.
        return a_out
    def toRasterFraction(arr_in, name_list):
        #path = os.path.join(params['DATA_CUBE_DIR'], tile, '{year}-{year}_001-365_HL_TSA_SEN2L_BLU_TSI.tif'.format(year=year))
        y1 = int(year)-3
        y2 = int(year)+1
        path = os.path.join(params['DATA_CUBE_DIR'], tile, '{y1}-{y2}_001-365_HL_TSA_SEN2L_BLU_RSP.tif'.format(y1=y1, y2=y2))
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        [cols, rows] = arr.shape
        bands = arr_in.shape[-1]
        driver = gdal.GetDriverByName("GTiff")
        path_out = path = os.path.join(params['PREDICTION_DIR'], tile, 'fraction_' + year + '.tif')
        print(path_out)
        outdata = driver.Create(path_out, rows, cols, bands, gdal.GDT_Byte)
        #outdata = driver.Create(path_out, rows, cols, bands, gdal.GDT_Float32)
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input
        for i in range(bands):
            outdata.GetRasterBand(i + 1).WriteArray(arr_in[..., i])
            outdata.GetRasterBand(i + 1).SetNoDataValue(params['NO_DATA_OUTPUT'])
            outdata.GetRasterBand(i + 1).SetDescription(name_list[i])
        outdata.FlushCache() ##saves to disk!!
        outdata = None
        band=None
        ds=None

    def toRasterSinglePrediction(arr_in, name_list, model_num):
        y1 = int(year)-3
        y2 = int(year)+1
        path = os.path.join(params['DATA_CUBE_DIR'], tile, '{y1}-{y2}_001-365_HL_TSA_SEN2L_BLU_RSP.tif'.format(y1=y1, y2=y2))
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        [cols, rows] = arr.shape
        bands = arr_in.shape[-1]
        driver = gdal.GetDriverByName("GTiff")
        path_out = path = os.path.join( '/data/ahsoka/eocp/klehr/RLP_SynthMix/42_avg_prediction/2023_new/5_single_prediction', tile, 'fraction_model' + str(model_num) + '.tif')
        if not os.path.exists(os.path.join( '/data/ahsoka/eocp/klehr/RLP_SynthMix/42_avg_prediction/2023_new/5_single_prediction', tile)):
                os.makedirs(os.path.join( '/data/ahsoka/eocp/klehr/RLP_SynthMix/42_avg_prediction/2023_new/5_single_prediction', tile))
        print(path_out)
        outdata = driver.Create(path_out, rows, cols, bands, gdal.GDT_Byte)
        #outdata = driver.Create(path_out, rows, cols, bands, gdal.GDT_Float32)
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input
        for i in range(bands):
            outdata.GetRasterBand(i + 1).WriteArray(arr_in[..., i])
            outdata.GetRasterBand(i + 1).SetNoDataValue(params['NO_DATA_OUTPUT'])
            outdata.GetRasterBand(i + 1).SetDescription(name_list[i])
        outdata.FlushCache() ##saves to disk!!
        outdata = None
        band=None
        ds=None

    def toRasterDeviation(arr_in, name_list):
        #path = os.path.join(params['DATA_CUBE_DIR'], tile, '{year}-{year}_001-365_HL_TSA_SEN2L_BLU_TSI.tif'.format(year=year))
        y1 = int(year)-3
        y2 = int(year)+1
        path = os.path.join(params['DATA_CUBE_DIR'], tile, '{y1}-{y2}_001-365_HL_TSA_SEN2L_BLU_RSP.tif'.format(y1=y1, y2=y2))
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        [cols, rows] = arr.shape
        bands = arr_in.shape[-1]
        driver = gdal.GetDriverByName("GTiff")
        path_out = path = os.path.join(params['PREDICTION_DIR'], tile, 'deviation_' + year + '.tif')
        print(path_out)
        outdata = driver.Create(path_out, rows, cols, bands, gdal.GDT_Float32)
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input
        for i in range(bands):
            outdata.GetRasterBand(i + 1).WriteArray(arr_in[..., i])
            outdata.GetRasterBand(i + 1).SetNoDataValue(params['NO_DATA_OUTPUT'])
            outdata.GetRasterBand(i + 1).SetDescription(name_list[i])
        outdata.FlushCache() ##saves to disk!!
        outdata = None
        band=None
        ds=None

    def toRasterDeviationRelative(arr_in, name_list):
        #path = os.path.join(params['DATA_CUBE_DIR'], tile, '{year}-{year}_001-365_HL_TSA_SEN2L_BLU_TSI.tif'.format(year=year))
        y1 = int(year)-3
        y2 = int(year)+1
        path = os.path.join(params['DATA_CUBE_DIR'], tile, '{y1}-{y2}_001-365_HL_TSA_SEN2L_BLU_RSP.tif'.format(y1=y1, y2=y2))
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        [cols, rows] = arr.shape
        bands = arr_in.shape[-1]
        driver = gdal.GetDriverByName("GTiff")
        path_out = path = os.path.join(params['PREDICTION_DIR'], tile, 'deviation_relative_' + year + '.tif')
        print(path_out)
        outdata = driver.Create(path_out, rows, cols, bands, gdal.GDT_Float32)
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input
        for i in range(bands):
            outdata.GetRasterBand(i + 1).WriteArray(arr_in[..., i])
            outdata.GetRasterBand(i + 1).SetNoDataValue(params['NO_DATA_OUTPUT'])
            outdata.GetRasterBand(i + 1).SetDescription(name_list[i])
        outdata.FlushCache() ##saves to disk!!
        outdata = None
        band=None
        ds=None

    def toRasterClassification(arr_in):
        #path = os.path.join(params['DATA_CUBE_DIR'], tile, '{year}-{year}_001-365_HL_TSA_SEN2L_BLU_TSI.tif'.format(year=year))
        y1 = int(year)-3
        y2 = int(year)+1
        path = os.path.join(params['DATA_CUBE_DIR'], tile, '{y1}-{y2}_001-365_HL_TSA_SEN2L_BLU_RSP.tif'.format(y1=y1, y2=y2))
        ds = gdal.Open(path)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        [cols, rows] = arr.shape
        driver = gdal.GetDriverByName("GTiff")
        path_out = os.path.join(params['PREDICTION_DIR'], tile, 'classification_' + year + '.tif')
        outdata = driver.Create(path_out, rows, cols, 1, gdal.GDT_Byte)
        outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(arr_in)
        outdata.GetRasterBand(1).SetNoDataValue(params['NO_DATA_OUTPUT'])
        # outdata.GetRasterBand(1).SetDescription()
        outdata.FlushCache() ##saves to disk!!
        outdata = None
        band=None
        ds=None

    # necesarry steps before processing
    y1 = int(year)-3 #!!!!!!!!! change to -3 again
    y2 = int(year)+1
    blue_band = os.path.join(params['DATA_CUBE_DIR'], tile, '{y1}-{y2}_001-365_HL_TSA_SEN2L_BLU_RSP.tif'.format(y1=y1, y2=y2))
    if not os.path.isfile(blue_band):
        print('Not tile, skipping!')
        return

    out_dir = os.path.join(params['PREDICTION_DIR'], tile)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # start processing here
    print('Predicting', tile, year, '...', str(no_of_tile) , '/37 ----------',sep = ' ')

    x_in = get_stack(tile, year)
    nodata_mask = x_in[:, :, 0, 0] == -9999
    if params['NORMALIZE_INPUT']:
        x_in = norm(x_in.astype(np.float32))
   
    y_out = np.zeros([x_in.shape[0], x_in.shape[1], len(params['TREE_CLASS_LIST'])]) 
    name_list = params['TREE_NAME_LIST']

    # multi model prediction
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
        # write single results
        # toRasterSinglePrediction(single, name_list, model_num)
        #print(tile + ': Model No. ' + str(model_num))

    stacked_arrays = np.stack(list_predictions, axis=-1)
    # avg fraction
    average_array = np.mean(stacked_arrays, axis=-1)

    # average daviation (careful!!!! adapt to number of models)
    deviation      = (np.absolute(stacked_arrays[...,0]-average_array) +
                      np.absolute(stacked_arrays[...,1]-average_array) +
                      np.absolute(stacked_arrays[...,2]-average_array) +
                      np.absolute(stacked_arrays[...,3]-average_array) +
                      np.absolute(stacked_arrays[...,4]-average_array) +
                      np.absolute(stacked_arrays[...,5]-average_array) +
                      np.absolute(stacked_arrays[...,6]-average_array) +
                      np.absolute(stacked_arrays[...,7]-average_array) +
                      np.absolute(stacked_arrays[...,8]-average_array) +
                      np.absolute(stacked_arrays[...,9]-average_array))/10
    # average deviation relative to the averaged fraction result
    #deviation_rel = deviation/average_array
    # take care for inf (x/0) and no data values(0/0) - real no daat values will be flaged later via no data mask
    #deviation_rel = np.nan_to_num(deviation_rel, neginf=0, posinf=0, nan = 0)

    # classification of dominant species
    y_out_clf = np.argmax(average_array, axis= -1)
    y_out_clf += 1
    y_out_clf[nodata_mask] = 255
    y_out_clf = y_out_clf.astype(np.int8)

    # writing raster
    average_array[nodata_mask] = 255
    #average_array = average_array.astype(np.int8)   
    toRasterFraction(average_array, name_list)
    
    deviation[nodata_mask] = 255
    #deviation = deviation.astype(np.int8)
    toRasterDeviation(deviation, name_list)
    
    #deviation_rel[nodata_mask] = 255
    #deviation_rel = deviation_rel.astype(np.int8)
    #toRasterDeviationRelative(deviation_rel, name_list)
    
    toRasterClassification(y_out_clf)
    print('-------- Predicting ' + tile + ' done successfully  | ' + str(no_of_tile) + '/37 ----------')

if __name__ == '__main__':
    
    model_list = []
    for i in range(params['NUM_MODELS']):
        model_path = os.path.join(params['SAVED_MODEL_PATH'],'version' +str(i+1)) 
        model = tf.keras.models.load_model(model_path)
        model_list.append(model)

    list_tile = os.listdir(params['DATA_CUBE_DIR'])
    list_year = params['YEAR_LIST']
    for year in list_year:
        Parallel(n_jobs=10)(delayed(predict)(tile, year, model_list, list_tile.index(tile)) for tile in list_tile)