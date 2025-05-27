import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random
import os
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--working_directoy", help="path to the pure data numpy array", default= "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/Synth_Mix/2021_ThermalTime")
parser.add_argument("--num_models", help="number of models you want to create", default= 2)
parser.add_argument("--year", help="year of synthetic mixture", default= '2021')
parser.add_argument("--tree_labels", help="labels of the tree species/classes in the correct order", default = "['Fichte','Kiefer','Tanne','Douglasie','Larche','Buche','Eiche','Ahorn','Birke','Erle','Pappel','Weide', 'Ground', 'Shadow']")

parser.add_argument("--num_hidden_layer", help="number of hidden layer", default= 5)
parser.add_argument("--hidden_layer_nodes", help="number of nodes per hidden layer", default = 128)
parser.add_argument("--learning_rate", help="learning_rate for training", default = 1e-3)
parser.add_argument("--batch_size", help="the batch size for training", default = 256)
parser.add_argument("--epochs", help="number of epochs", default = 5)


args = parser.parse_args()

def train(model_number):
    
    def norm(a):
        a_out = a/10000.
        return a_out
    
    #------------------------------ added --------------------------------
    class SumToOneLayer(tf.keras.layers.Layer):
            def call(self, inputs):
                return inputs / tf.reduce_sum(inputs, axis=-1, keepdims=True)
    #---------------------------------------------------------------------
    
    def get_model(input_shape, lc_num, hidden_layer_num, hidden_layer_node):
        def dense(x, filter_size):
            layer = tf.keras.layers.Dense(filter_size)(x)
            return layer
        x_in = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Flatten()(x_in)
        for _ in range(hidden_layer_num):
            #x = tf.nn.relu(dense(x, hidden_layer_node))
            x = tf.keras.layers.ReLU()(dense(x, hidden_layer_node))
        x_out = dense(x, lc_num)
        # with sum to one condition
        x_out = SumToOneLayer()(x_out)
        model =tf.keras.Model(inputs = x_in, outputs = x_out)
        return model
    
    def get_loss(x, y, model, training=True):
        y_pred = model(x, training=training)
        loss = tf.keras.losses.MeanAbsoluteError()(y, y_pred)
        return loss

    @tf.function
    def train(x, y, model, opt):
        with tf.GradientTape() as tape:
            loss = get_loss(x, y, model, training=True)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    

    x_train = []
    y_train = []
    # load the synthetic mixed data
    print("trained model " + str(model_number))
    x_mixed_out_path = os.path.join(args.working_directoy, '2_mixed_data','version' +str(model_number) , 'x_mixed_' + str(args.year) + '.npy')
    y_mixed_out_path = os.path.join(args.working_directoy, '2_mixed_data','version' +str(model_number) , 'y_mixed_' + str(args.year) + '.npy')
    x_train.append(np.load(x_mixed_out_path))
    y_train.append(np.load(y_mixed_out_path))
    # prepare synthetic mixed data
    x_train = np.concatenate(x_train, axis=0, dtype=np.float32)
    x_train = norm(x_train)
    y_train = np.concatenate(y_train, axis=0)
    
    # define parameter
    input_shape = (x_train.shape[1], x_train.shape[2])
    lc_num = len(  ast.literal_eval(args.tree_labels))
    hidden_layer_num = args.num_hidden_layer
    hidden_layer_node = args.hidden_layer_nodes
    
    model = get_model(input_shape, lc_num, hidden_layer_num, hidden_layer_node)
    lr = float(args.learning_rate)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    train_index = list(range(y_train.shape[0]))
    
    batch_size = int(args.batch_size)
    iterations = int(y_train.shape[0]/batch_size)
    epochs = int(args.epochs)
    random.shuffle(train_index)

    if not os.path.exists( os.path.join(args.working_directoy, '3_trained_model_test' ,'version' +str(model_number))):
        os.makedirs( os.path.join(args.working_directoy, '3_trained_model_test' ,'version' +str(model_number)))

    with open(os.path.join(args.working_directoy, '3_trained_model_test' ,'version' +str(model_number),'performance.txt'), 'w') as file:
        file.write(f"'Epoch'; 'MAE' \n")

    for e in range(epochs):
        loss_train = 0
        for i in tqdm(range(iterations)):
            x_batch = x_train[train_index[i*batch_size: i*batch_size + batch_size]]
            y_batch = y_train[train_index[i*batch_size: i*batch_size + batch_size]]
            loss_train += train(x_batch, y_batch, model, opt)
        
        loss_train = loss_train / iterations
        loss_train = loss_train.numpy()
        
        print('Epoch: ', e)
        print('MAE: ', loss_train)
        with open(os.path.join(args.working_directoy, '3_trained_model_test','version' + str(model_number),'performance.txt'), 'a') as file:
            file.write(f"{e};{loss_train}\n")
        random.shuffle(train_index)
        #lr *= params['LEARNING_RATE_DECAY']
        opt.learning_raye = lr

    model_path = os.path.join(args.working_directoy, '3_trained_model_test','version' + str(model_number), 'saved_model'+ str(model_number)+ '.keras')
    tf.keras.models.save_model(model, model_path)
    print('Model is saved at ', model_path)

if __name__ == '__main__':
    for i in range(int(args.num_models)):
        train(i+1+2)