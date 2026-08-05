import numpy as np
import random
from tqdm import tqdm
import os
import argparse
import ast
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the pure data numpy array", 
                    #default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix/2018_canopy_gap_fraction")
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix_CanopyGap")
parser.add_argument("--year", help="year of synthetic mixture", default= '2025')
parser.add_argument("--num_libs", help="number of synthtic libraries to create", default= 10)
parser.add_argument("--lib_size", help="number of synthtic libraries to create", default= 256000)
parser.add_argument("--tree_index", help="labels of the tree species/classes in the correct order", 
                    default = '[1,2,3,4,5,6,7,8,9,10,11,12,13]')
parser.add_argument("--tree_class_weights", help="labels of the tree species/classes in the correct order", 
                    #default = '[1,1,1,1,1, 1,1,1,1,1, 1,1,1,1]')
                    default = '[1,1,1,1,1, 1,1,1,1,1, 1,1,1]') # no poplar
parser.add_argument("--mixture_list", help="list of mixing complexity - how many classes can be mixed in one mixture", 
                    default = '[1,2,3]' )
parser.add_argument("--mixture_weights", 
                    help="wheight for every mixing complexity - For example [1, 1, 5, 1] will increase more chances to have 3-class mixtures", 
                    default = '[1, 5, 5]' )
args = parser.parse_args()

def make_array(folder):
    files = []
    for datei in os.listdir(folder):
        files.append(os.path.join(folder,datei))
    files = sorted(files)

    array_list = []
    for file in files:
        data = np.loadtxt(file, delimiter=",")
        array_list.append(data)
    array = np.stack(array_list)
    return(array)

def mixing(year,model_number):
    print('version ' + str(model_number))
    x_pure = make_array(os.path.join(args.working_directory, args.year, '1_pure' , f'samples_x{str(args.year)}'))
    x_pure = x_pure.astype(np.float32)
    y_pure = make_array(os.path.join(args.working_directory, args.year, '1_pure' , f'samples_y{str(args.year)}'))

    # check about nodata values
    mask = np.any(x_pure == -9999, axis=1)
    x_pure = x_pure[~mask]
    y_pure = y_pure[~mask]

    #y_pure_species = y_pure[np.isin(y_pure,[1,2,3,4,5,6,7,8,9,10,11,12,14])] # ground -> own class
    #x_pure_species = x_pure[np.isin(y_pure,[1,2,3,4,5,6,7,8,9,10,11,12,14]),:] 
    # Poplar out
    y_pure_species = y_pure[np.isin(y_pure,[1,2,3,4,5,6,7,8,9,10,11,13])] # Class 12 = ground -> own class
    x_pure_species = x_pure[np.isin(y_pure,[1,2,3,4,5,6,7,8,9,10,11,13]),:] 

    y_pure_shadow = y_pure[np.isin(y_pure,[13])]     # shadow class
    x_pure_shadow = x_pure[np.isin(y_pure,[13]),:] 
    y_pure_ground = y_pure[np.isin(y_pure,[12])]     # ground class
    x_pure_ground = x_pure[np.isin(y_pure,[12]),:]

    #------------------------------------------------------------
    # perform the mixing
    #------------------------------------------------------------
    training_sample = int(args.lib_size)
    x_mixed = []
    y_mixed = []
    index_list = list(range(len(y_pure)))
    lc_index = ast.literal_eval(args.tree_index)

    for _ in tqdm(range(training_sample)):
        k = random.choices(ast.literal_eval(args.mixture_list), 
                           weights= ast.literal_eval(args.mixture_weights), k=1 )[0]
        fractions = np.random.dirichlet(np.ones(k),size=1)[0]
        chosen_classes = random.choices(lc_index, k=k, weights= ast.literal_eval(args.tree_class_weights))

        x = 0
        y = np.zeros(len(lc_index))

        for i in range(len(chosen_classes)):
            #if chosen_classes[i] == 13: # new canopy gap class
            if chosen_classes[i] == 12: # new canopy gap class
                #print('mixing canopy gap class')
                bg_fraction = np.random.dirichlet(np.ones(2),size=1)[0]
                chosen_index_shadow = random.sample(list(range(len(y_pure_shadow))), k=1)
                chosen_index_ground = random.sample(list(range(len(y_pure_ground))), k=1)
                ts = x_pure_shadow[chosen_index_shadow]*bg_fraction[0] + x_pure_ground[chosen_index_ground]*bg_fraction[1]
                x += ts[0,:]*fractions[i]
            else:
                species_indices = np.where(y_pure_species == chosen_classes[i])[0]
                chosen_species_index = random.choice(species_indices)
                x += x_pure_species[chosen_species_index]*fractions[i]
            label_pos = lc_index.index(chosen_classes[i])
            y[label_pos] += fractions[i]
        x_mixed.append(x)
        y_mixed.append(y)

    x_mixed = np.array(x_mixed, np.int16)
    y_mixed = np.array(y_mixed, np.float32)

    y_mixed = np.multiply(y_mixed, 100)
    y_mixed_int = y_mixed.astype(np.int16)
    #------------------------------------------------------------
    #                   store the mixed data
    #------------------------------------------------------------

    os.makedirs(os.path.join(args.working_directory, args.year, '2_mixed_data_glob','version' +str(model_number)), exist_ok=True)
    x_mixed_out_path = os.path.join(args.working_directory, args.year, '2_mixed_data_glob','version' +str(model_number), 'x_mixed_' + str(year) + '.npz')
    y_mixed_out_path = os.path.join(args.working_directory, args.year, '2_mixed_data_glob','version' +str(model_number), 'y_mixed_' + str(year) + '.npz')
    np.savez_compressed(x_mixed_out_path, x_mixed=x_mixed)
    np.savez_compressed(y_mixed_out_path, y_mixed=y_mixed_int)


if __name__ == '__main__':
    for i in range(int(args.num_libs)):
        mixing(args.year,i+1+10)