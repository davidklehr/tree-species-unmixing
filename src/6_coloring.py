import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from joblib import Parallel, delayed

from joblib import Parallel, delayed
# start pytohn code
import rasterio
import numpy as np
import colorsys
from tqdm import tqdm
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the pure data numpy array", 
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix_CanopyGap")
parser.add_argument("--year", help="number of models you want to create", default= '2018')
parser.add_argument("--tree_labels", help="labels of the tree species/classes in the correct order", 
                    #default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Ahorn','Birke','Erle','Pappel','OtherDT', 'Canopy Gap']")
                    default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Ahorn','Birke','Erle','OtherDT', 'Canopy Gap']")
parser.add_argument("--tile", help="The tile to be color", default= 'X0055_Y0053')
args = parser.parse_args()

def hsv_to_rgb(h, s, v):
    # Umwandlung auf den [0,1] Bereich für Hue
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return r, g, b

# Pfad zur GeoTIFF-Datei


bands = ast.literal_eval(args.tree_labels)

def color_raster(working_dir, tile, no_of_tile, length_list):

    print('-------- Start coloring ' + tile + ' | ' + str(no_of_tile) + f'/{str(length_list)} ----------')
    if not tile.startswith('X'):
       print('Not tile, skipping!')
       return

    tiff_path = os.path.join(working_dir, tile , 'tree_fraction_and_gap.tif')
    tiff_path
    # Definiere das Ausgabe-GeoTIFF
    output_tiff_path = os.path.join(working_dir, tile , 'HSV_colored.tif')

    if not os.path.isfile(os.path.join(working_dir, tile,'tree_fraction_and_gap.tif')):
        print('folder ' + tile + 'is empty.')
        return

    with rasterio.open(tiff_path) as src:

        # Meta-Daten
        crs = src.crs
        transform = src.transform
        profile = src.profile

        # Als numpy array laden (alle Bänder)
        descriptions = src.descriptions

        #nodatamask
        mask = src.read(1)
          
        # OtherDT
        #OtherDT_band_indexes = [i+1 for i, desc in enumerate(descriptions) if desc in ['Birch', 'Willow', 'Robinia', 'Poplar']]
        #OtherDT_band_indexes = [i+1 for i, desc in enumerate(descriptions) if desc in ['Birke', 'Weide', 'Pappel']]
        #OtherDT_band_indexes = [i+1 for i, desc in enumerate(descriptions) if desc in ['Pappel', 'OtherDT']]
        OtherDT_band_indexes = [i+1 for i, desc in enumerate(descriptions) if desc in [ 'OtherDT']]
        selected_bands = [src.read(i) for i in OtherDT_band_indexes]
        #print(np.stack(selected_bands, axis =0).shape)
        Other_DT = np.sum(selected_bands, axis=0)  # shape: (height, width)
        #print(Other_DT.shape)
        Other_DT_expanded = Other_DT[np.newaxis, :, :]

        # Background
        background_band_indexes = [i+1 for i, desc in enumerate(descriptions) if desc in ['Canopy Gap']]
        selected_bands = [src.read(i) for i in background_band_indexes]
        #print(np.stack(selected_bands, axis =0).shape)
        background = np.sum(selected_bands, axis=0)  # shape: (height, width)

        #Array
        #band_indexes = [i+1 for i, desc in enumerate(descriptions) if desc in  ['Beech', 'Oak', 'Alder', 'Maple', 'Pine', 'Spruce', 'Douglas', 'Larch']]
        band_indexes = [i+1 for i, desc in enumerate(descriptions) if desc in  ['Buche', 'Eiche','Birke','Erle', 'Ahorn', 'Kiefer', 'Fichte', 'Douglasie', 'Larche', 'Tanne']]
        selected_bands = [src.read(i) for i in band_indexes]
        data = np.stack(selected_bands, axis =0)
        #print(data.shape)

        # Anhängen entlang der Band-Achse (axis=0)
        data = np.concatenate((data, Other_DT_expanded), axis=0)
        #print(data.shape)
        # Maximalwert je Pixel (über axis=0, also über die Bänder)
        max_values = np.max(data, axis=0)  # shape: (3000, 3000)
        # Index des Bandes mit dem Maximalwert je Pixel
        max_indices = np.argmax(data, axis=0)

        #---------------------------
        # HSV array
        value = 1 - (background / 100)
        saturation = max_values / 100
        #hue_map = {0: 120, 1: 60, 2: 190, 3: 25, 
        #            4: 0, 5: 240, 6: 320, 7: 45, 8: 160}
        hue_map = {0: 240,   1:0  , 2:280 ,   3:320   ,    4: 45  , 5:120 , 6:60  ,    7:25   ,  8: 210  ,   9:180  , 10: 160}  
                #  Fichte,  Kiefer| Tanne | Douglasie |  Lärche   | Buche | Eiche | Ahorn     |   Birke  | Erle     | otherDT
                #   blau |    rot | lila  |    pink   | h. orange |  grün | gelb  | d. orange | m.  blau | hellblau | türkis
        hue = np.array([[hue_map[idx] for idx in row] for row in max_indices])

        hsv_array = np.stack([hue, saturation, value], axis=0)

        rgb_array = np.zeros_like(hsv_array)
        #for i in tqdm(range(hsv_array.shape[1])):
        for i in range(hsv_array.shape[1]):
            for j in range(hsv_array.shape[2]):
                h, s, v = hsv_array[:, i, j]  # Hole Hue, Saturation und Value
                r, g, b = hsv_to_rgb(h, s, v)
               
                rgb_array[0, i, j] = int(r*254)  # Red-Kanal
                rgb_array[1, i, j] = int(g*254)  # Green-Kanal
                rgb_array[2, i, j] = int(b*254)  # Blue-Kanal
        # no data assignment
        rgb_array[0][mask==255]=255
        rgb_array[1][mask==255]=255
        rgb_array[2][mask==255]=255

        with rasterio.open(output_tiff_path, 'w',  driver='GTiff', nodata=255,
                    height=rgb_array.shape[1], width=rgb_array.shape[2], 
                    count=3, dtype='uint8', crs=crs, transform=transform, compress="zstd") as dst:
            dst.write(rgb_array[0], 1)  # Rot-Kanal in Band 1
            dst.write(rgb_array[1], 2)  # Grün-Kanal in Band 2
            dst.write(rgb_array[2], 3)  # Blau-Kanal in Band 3

    print('-------- Coloring ' + tile + ' done successfully  | ' + str(no_of_tile) + f'/{str(length_list)} ----------')


if __name__ == '__main__':
    #parallel --line-buffer -j 32 python -u 6_coloring.py --year 2018 --tile {} :::: /data/ahsoka/eocp/forestpulse/02_scripts/DWD/RLP_tilelist.txt
    working_dir = os.path.join(args.working_directory, args.year ,'5_prediction_distributed_glob')
    # color just one tile
    color_raster(working_dir, args.tile.strip(), 0, 1)
    #----------------------------
    