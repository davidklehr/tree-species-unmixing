from osgeo import gdal
from params_avg import params
import os

def GetExtent(ds):
    """ Return list of corner coordinates from a gdal Dataset """
    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel

    extent= [xmin, ymin, xmax, ymax]
    extent_string = ''
    for i in range(len(extent)):
        if i < len(extent) - 1:
            extent_string += str(extent[i]) + ' '
        else:
            extent_string += str(extent[i])
    return extent_string

if __name__ == '__main__':
    list_tile = os.listdir(params['DATA_CUBE_DIR'])
    for tile in list_tile:
        if tile == "mosaic":
            print('mosiac skipped')
            continue        
        print(tile)
        # Using blue band in 2022 to get the extent, change band or year if not available.

        # original:
        # blue_band_path = os.path.join(params['DATA_CUBE_DIR'], tile, '2017-2021_001-365_HL_TSA_SEN2L_BLU_RSP.tif')
        first_year = int(params['YEAR_LIST'][0])-3
        last_year = int(params['YEAR_LIST'][0])+1
        blue_band_path = os.path.join(params['DATA_CUBE_DIR'], tile, '{y1}-{y2}_001-365_HL_TSA_SEN2L_BLU_RSP.tif'.format(y1=first_year, y2=last_year))
        if not os.path.isfile(blue_band_path):
            print('Not tile, skipping!')
            print('Not tile, skipping!')
            continue
        
        out_dir = os.path.join(params['RASTERIZED_POINT_DIR'], tile)
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        point_out = os.path.join(out_dir, 'point_rasterized.tif')
        ds=gdal.Open(blue_band_path)
        extent=GetExtent(ds)
        rasterize_command = 'gdal_rasterize ' + \
                            '-a {} '.format(params['TREE_CLASS_COLUM_NAME']) + \
                            '-ts {pixel_num} {pixel_num} '.format(pixel_num = params['RASTER_PIXEL_NUM']) + \
                            '-a_nodata {} '.format(params['NO_DATA_OUTPUT']) + \
                            '-te {} '.format(extent) + \
                            '-ot Byte ' + \
                            '-of GTiff {point_path_in} {point_path_out}'.format(point_path_in=params['PURE_POINTS_PATH'], point_path_out=point_out)
        os.system(rasterize_command)

