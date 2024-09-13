import os
import rasterio
import numpy as np
from params_avg import params

def get_stack(tile, y1,y2):
    def get_band(band_name):
        band_path = os.path.join(params['DATA_CUBE_DIR'], tile, '{y1}-{y2}_001-365_HL_TSA_SEN2L_{band_name}_RSP.tif'.format(y1=y1, y2=y2,band_name=band_name))
        with rasterio.open(band_path) as src:
            band = src.read()
        band = np.moveaxis(band, 0, -1)
        return band
    band_list = params['BAND_LIST']
    stack = np.array([get_band(b) for b in band_list])
    stack = np.moveaxis(stack, 0, -1)
    return stack

if __name__ == '__main__':
    list_tile = os.listdir(params['DATA_CUBE_DIR'])
    y1 = int(params['YEAR_LIST'][0])-3
    y2 = int(params['YEAR_LIST'][0])+1
    
    for year in params['YEAR_LIST']:
        print(year, end = ' ', flush=True)
        x_list = []
        y_list = []
        
        for tile in list_tile:
            if tile == "mosaic":
                print('mosiac skipped')
                continue    
            print(tile, end = ' ', flush=True)
            rasterized_point_path = os.path.join(params['RASTERIZED_POINT_DIR'], tile, 'point_rasterized.tif')
            
            with rasterio.open(rasterized_point_path) as src:
                points = src.read(1)
                
            
            blue_band_path = os.path.join(params['DATA_CUBE_DIR'], tile, '{y1}-{y2}_001-365_HL_TSA_SEN2L_BLU_RSP.tif'.format(y1=y1 , y2=y2))
            if not os.path.isfile(blue_band_path):
                print('Not tile, skipping!')
                print(blue_band_path)
                continue
            
            stack = get_stack(tile, y1,y2)
            
            for lc in params['TREE_CLASS_LIST']:
                mask = points == lc
                if np.sum(mask) < 1:
                    continue
                x = stack[mask, ...]
                y = np.zeros(np.sum(mask), np.int8) + lc
                x_list.append(x)
                y_list.append(y)
        x_list = np.concatenate(x_list, 0)
        y_list = np.concatenate(y_list, 0)
        x_list_out_path = os.path.join(params['EXTRACTED_SPECTRA_DIR'], 'x' + year + '.npy')
        y_list_out_path = os.path.join(params['EXTRACTED_SPECTRA_DIR'], 'y' + year + '.npy')
        np.save(x_list_out_path, arr=x_list)
        np.save(y_list_out_path, arr=y_list)
        print('Done')