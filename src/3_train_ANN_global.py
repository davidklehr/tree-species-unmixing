from matplotlib import cm
import tensorflow as tf
from keras.saving import register_keras_serializable
import numpy as np
from tqdm import tqdm
import random
import os
import argparse
import ast
from joblib import Parallel, delayed
import time
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--working_directory", help="path to the pure data numpy array", 
                    #default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix/2018_canopy_gap_fraction")
                    default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix_CanopyGap")
parser.add_argument("--num_models", help="number of models you want to create", default= 10)
parser.add_argument("--year", help="year of synthetic mixture", default= '2025')
parser.add_argument("--tree_labels", help="labels of the tree species/classes in the correct order", 
                    #default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Ahorn','Birke','Erle','Pappel','OtherDT', 'Gap Fraction', 'Shadow']")
                    default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Ahorn','Birke','Erle','OtherDT', 'Gap Fraction', 'Shadow']")
parser.add_argument("--learning_rate", help="learning_rate for training", default = 1e-3)
parser.add_argument("--batch_size", help="the batch size for training", default = 256) # orig 256
parser.add_argument("--epochs", help="number of epochs", default = 125)
args = parser.parse_args()

def train(model_number, pos):
    #------------------------------ added --------------------------------
    @register_keras_serializable(package="Custom")
    class SumToOneLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            return inputs / tf.reduce_sum(inputs, axis=-1, keepdims=True)
    #---------------------------------------------------------------------
    
    def norm(a):
        a_out = a/10000.
        return a_out
    
    def get_model(input_shape, lc_num):
        def dense(x, filter_size):
            layer = tf.keras.layers.Dense(filter_size)(x)
            return layer
        
        # define input layer
        x_in = tf.keras.Input(shape=(input_shape,))
        x = x_in 
        static_in = tf.keras.Input(shape=(1,))
        x = tf.keras.layers.Concatenate()([x, static_in])

        # architecture of hidden layers
        x = tf.keras.layers.ReLU()(dense(x, 128))
        x = tf.keras.layers.ReLU()(dense(x, 256))
        x = tf.keras.layers.ReLU()(dense(x, 512))
        x = tf.keras.layers.ReLU()(dense(x, 256))
        x = tf.keras.layers.ReLU()(dense(x, 128))

        # define output
        x_out = dense(x, lc_num)
        x_out = SumToOneLayer()(x_out)
        
        model =tf.keras.Model(inputs = [x_in, static_in], outputs = x_out)
        #print(model.summary())
        return model
    
    def get_loss(x, y, model, training=True):
        y_pred = model(x, training=training)
        loss = tf.keras.losses.MeanAbsoluteError()(y, y_pred)
        return loss

    @tf.function
    def train(x_seq, x_static, y, model, opt):
        with tf.GradientTape() as tape:
            loss = get_loss([x_seq, x_static], y, model, training=True)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    
    # -----------------------------
    # 1. load the synthetic mixed data
    #------------------------------
    x_mixed_out_path = os.path.join(args.working_directory, args.year, '2_mixed_data_glob','version' +str(model_number) , 'x_mixed_' + str(args.year) + '.npz')
    y_mixed_out_path = os.path.join(args.working_directory, args.year, '2_mixed_data_glob','version' +str(model_number) , 'y_mixed_' + str(args.year) + '.npz')
    x_train = np.load(x_mixed_out_path)
    x_train = x_train['x_mixed'].astype(np.float32)
    y_train = np.load(y_mixed_out_path)
    y_train = y_train['y_mixed'].astype(np.float32)
    y_train = y_train / 100.

    x_train = norm(x_train)

    ground_fraction = y_train[:,-2]  # assuming canpoy gap is the second last class
    ground_fraction = ground_fraction.reshape(-1, 1)

    # -----------------------------
    # 2. define ANN and training parameter
    # -----------------------------
    # 2.1 model parameter
    input_shape = (x_train.shape[1])
    lc_num = len(  ast.literal_eval(args.tree_labels))
    model = get_model(input_shape, lc_num)

    #2.2 training parameter
    lr = float(args.learning_rate)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    train_index = list(range(y_train.shape[0]))
    batch_size = int(args.batch_size)
    iterations = int(y_train.shape[0]/batch_size)
    epochs = int(args.epochs)
    random.shuffle(train_index)

    if not os.path.exists( os.path.join(args.working_directory, args.year, '3_trained_model_glob','version' +str(model_number))):
        os.makedirs( os.path.join(args.working_directory, args.year, '3_trained_model_glob' ,'version' +str(model_number)))

    with open(os.path.join(args.working_directory, args.year, '3_trained_model_glob' ,'version' +str(model_number),'performance.txt'), 'w') as file:
        file.write(f"Epoch;MAE;time\n")

    # -----------------------------
    # 3. Training 
    # -----------------------------
    start_time = time.time()
    pbar = tqdm(total=epochs, desc=f"Model {model_number}", position=pos, leave=True)
    for e in range(epochs):
        loss_train = 0
        for i in range(iterations):
            idx = train_index[i*batch_size : i*batch_size + batch_size]
            x_batch = x_train[idx]
            y_batch = y_train[idx]
            x_static_batch = ground_fraction[idx]
            loss_train += train(x_batch, x_static_batch, y_batch, model, opt)

        loss_train = loss_train / iterations
        loss_train = loss_train.numpy()
    
        passed_time_min = int((time.time() - start_time) // 60)
        passed_time_sec = (time.time() - start_time) - (passed_time_min * 60)
        with open(os.path.join(args.working_directory, args.year, '3_trained_model_glob','version' + str(model_number),'performance.txt'), 'a') as file:
            file.write(f"{e};{loss_train};{str(passed_time_min)}:{str(int(round(passed_time_sec,0))) }\n")
        random.shuffle(train_index)
        if e > 30: #vllt besser 100
            opt.learning_rate = lr/10
            if e > 75: # dann 140; 150 Ende
                opt.learning_rate = lr/100
                if e > 100: # 
                    opt.learning_rate = lr/1000

        pbar.update(1)
        pbar.set_description(f"Model {model_number} | Epoch {e+1} | MAE={loss_train:.4f}")
    pbar.close()

    # save the trained model
    model_path = os.path.join(args.working_directory, args.year, '3_trained_model_glob','version' + str(model_number), 'saved_model'+ str(model_number)+ '.keras')
    tf.keras.models.save_model(model, model_path)
    return
                    
def plot():
    versions = os.listdir(os.path.join(args.working_directory, args.year, '3_trained_model_glob' ))
    files = [os.path.join(args.working_directory, args.year, '3_trained_model_glob', version , 'performance.txt') for version in sorted(versions) if version.startswith('version')]
    # --- Farben vorbereiten (10 Blautöne) ---
    cmap = plt.get_cmap('Blues', len(files) + 3)
    colors = [cmap(i + 3) for i in range(len(files))]  # überspringe die sehr hellen

    # --- Plot erstellen ---
    plt.figure(figsize=(9, 5))

    for i, (file, color) in enumerate(zip(files, colors), start=1):
        df = pd.read_csv(file, sep=';', engine='python', header=0)
        plt.plot(df['Epoch'], df['MAE'],
             label=f"Modell {i}",
             marker='o', linestyle='-', markersize=3,
             color=color,
             linewidth=1.8)
    plt.title("MAE-Verlauf aller Modelle")
    plt.xlabel("Epoche")
    plt.ylabel("MAE")
    # Achsenbereiche & Gitter
    plt.xlim(-5, int(args.epochs))
    plt.ylim(0, 0.12)
    plt.yticks(np.arange(0, 0.12, 0.02))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Legende rechts neben dem Plot
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="Modelle")
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Platz für Legende rechts

    # --- Speichern ---
    plt.savefig(os.path.join(args.working_directory, args.year, '3_trained_model_glob', 'mae_plot.png'), dpi=300)
    plt.close()

if __name__ == '__main__':
    num_workers = 10
    ####Parallel(n_jobs=num_workers, backend="loky")(delayed(train)(i+1, (i%num_workers)*2) for i in range(int(args.num_models)))
    Parallel(n_jobs=num_workers, backend="loky")(delayed(train)(i+1+10, i) for i in range(int(args.num_models)))
    plot()