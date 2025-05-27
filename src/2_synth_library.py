import numpy as np
import random
from tqdm import tqdm
import os
from params_avg2 import params

def mixing(year,model_number):
    print(year)
    print('version ' + str(model_number))
    x_pure_path = os.path.join(params['EXTRACTED_SPECTRA_DIR'], 'x' + year + '.npy')
    y_pure_path = os.path.join(params['EXTRACTED_SPECTRA_DIR'], 'y' + year + '.npy')
    x_pure = np.load(x_pure_path)
    x_pure = x_pure.astype(np.float32)
    y_pure = np.load(y_pure_path)
    training_sample = params['MIXED_NUM']
    x_mixed = []
    y_mixed = []
    index_list = list(range(len(y_pure)))
    lc_index = params['TREE_CLASS_LIST']
    for _ in tqdm(range(training_sample)):
        k = random.choices(params['MIXTURE_LIST'], weights=params['MIXTURE_WEIGHTS'], k=1)[0]
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
    if not os.path.exists(os.path.join(params['MIXED_SPECTRA_DIR'],'version' +str(model_number))):
            os.makedirs(os.path.join(params['MIXED_SPECTRA_DIR'],'version' +str(model_number)))
    x_mixed_out_path = os.path.join(params['MIXED_SPECTRA_DIR'],'version' +str(model_number), 'x_mixed_' + year + '.npy')
    y_mixed_out_path = os.path.join(params['MIXED_SPECTRA_DIR'],'version' +str(model_number), 'y_mixed_' + year + '.npy')
    print(x_mixed_out_path)
    np.save(x_mixed_out_path, arr=x_mixed)
    np.save(y_mixed_out_path, arr=y_mixed)

if __name__ == '__main__':
    for year in params['YEAR_LIST']:
        for i in range(params['NUM_MODELS']):
            mixing(year,i+1+1)