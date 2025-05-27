import numpy as np
import random
from tqdm import tqdm
import os
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--working_directoy", help="path to the pure data numpy array", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix/2021_ThermalTime")
parser.add_argument("--year", help="year of synthetic mixture", default= 2021)
parser.add_argument("--num_libs", help="number of synthtic libraries to create", default= 10)
parser.add_argument("--lib_size", help="number of synthtic libraries to create", default= 100)
#parser.add_argument("--tree_labels", help="labels of the tree species/classes in the correct order", default = ['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Ahorn','Birke','Erle','Pappel','Weide', 'Ground', 'Shadow'])
parser.add_argument("--tree_index", help="labels of the tree species/classes in the correct order", default = '[1,2,3,4,5,6,7,8,9,10,11,12,13,14]')
parser.add_argument("--mixture_list", help="list of mixing complexity - how many classes can be mixed in one mixture", default = '[1,2,3]' )
parser.add_argument("--mixture_weights", help="wheight for every mixing complexity - For example [1, 1, 5, 1] will increase more chances to have 3-class mixtures", default = '[1, 5, 5]' )
args = parser.parse_args()


def mixing(year,model_number):
    print(year)
    print('version ' + str(model_number))
    x_pure_path = os.path.join(args.working_directoy, '01_pure' ,f'x_{year}.npy')
    y_pure_path = os.path.join(args.working_directoy, '01_pure' , f'y_{year}.npy')
    x_pure = np.load(x_pure_path)
    x_pure = x_pure.astype(np.float32)
    y_pure = np.load(y_pure_path)
    training_sample = int(args.lib_size)
    x_mixed = []
    y_mixed = []
    index_list = list(range(len(y_pure)))
    lc_index = ast.literal_eval(args.tree_index)
    for _ in tqdm(range(training_sample)):
        k = random.choices(ast.literal_eval(args.mixture_list), 
                           weights= ast.literal_eval(args.mixture_weights), k=1 )[0]
        fractions = np.random.dirichlet(np.ones(k),size=1)[0]
        chosen_index = random.sample(index_list, k=k)
        x = 0
        y = np.zeros(len(lc_index))
        for i in range(len(chosen_index)):
            x += x_pure[chosen_index[i]]*fractions[i]
            label_pos = lc_index.index(y_pure[chosen_index[i]])
            y[label_pos] += fractions[i]
        x_mixed.append(x)
        y_mixed.append(y)
    x_mixed = np.array(x_mixed, np.float32)
    y_mixed = np.array(y_mixed, np.float32)
    if not os.path.exists(os.path.join(args.working_directoy, '2_mixed_data' ,'version' +str(model_number))):
            os.makedirs(os.path.join(args.working_directoy, '2_mixed_data','version' +str(model_number)))
    x_mixed_out_path = os.path.join(args.working_directoy, '2_mixed_data','version' +str(model_number), 'x_mixed_' + str(year) + '.npy')
    y_mixed_out_path = os.path.join(args.working_directoy, '2_mixed_data','version' +str(model_number), 'y_mixed_' + str(year) + '.npy')
    print(x_mixed_out_path)
    np.save(x_mixed_out_path, arr=x_mixed)
    np.save(y_mixed_out_path, arr=y_mixed)

if __name__ == '__main__':
    for i in range(int(args.num_libs)):
        mixing(args.year,i+1)