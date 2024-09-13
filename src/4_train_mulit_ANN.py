import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random
import os
from params_avg2 import params

def train(model_number):
    
    def norm(a):
        a_out = a/10000.
        return a_out
    
    def get_model(input_shape, lc_num, hidden_layer_num, hidden_layer_node):
        def dense(x, filter_size):
            layer = tf.keras.layers.Dense(filter_size)(x)
            return layer
        x_in = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Flatten()(x_in)
        for _ in range(hidden_layer_num):
            x = tf.nn.relu(dense(x, hidden_layer_node))
        x_out = dense(x, lc_num)
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
    

    year_list = params['YEAR_LIST']
    x_train = []
    y_train = []
    # load the synthetic mixed data
    print("trained model " +str(model_number))
    for year in year_list:
        x_mixed_out_path = os.path.join(params['MIXED_SPECTRA_DIR'],'version' +str(model_number) , 'x_mixed_' + year + '.npy')
        y_mixed_out_path = os.path.join(params['MIXED_SPECTRA_DIR'],'version' +str(model_number) , 'y_mixed_' + year + '.npy')
        x_train.append(np.load(x_mixed_out_path))
        y_train.append(np.load(y_mixed_out_path))
    # prepare synthetic mixed data
    x_train = np.concatenate(x_train, axis=0, dtype=np.float32)
    if params['NORMALIZE_INPUT']:
        x_train = norm(x_train)
    y_train = np.concatenate(y_train, axis=0)
    
    # define parameter
    input_shape = (x_train.shape[1], x_train.shape[2])
    lc_num = len(params['TREE_CLASS_LIST'])
    hidden_layer_num = params['NN_hidden_layer_num']
    hidden_layer_node = params['NN_hidden_layer_nodes']
    
    model = get_model(input_shape, lc_num, hidden_layer_num, hidden_layer_node)
    lr = params['LEARNING_RATE']
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    train_index = list(range(y_train.shape[0]))
    
    batch_size = params['TRAINING_BATCH_SIZE']
    
    iterations = int(y_train.shape[0]/batch_size)
    
    epochs = params['EPOCHS']
    random.shuffle(train_index)
    
    for e in range(epochs):
        loss_train = 0
       # print(tf.config.threading.get_intra_op_parallelism_threads())
       # print(tf.config.threading.get_inter_op_parallelism_threads())
        for i in tqdm(range(iterations)):
            x_batch = x_train[train_index[i*batch_size: i*batch_size + batch_size]]
            y_batch = y_train[train_index[i*batch_size: i*batch_size + batch_size]]
            loss_train += train(x_batch, y_batch, model, opt)
        
        loss_train = loss_train / iterations
        loss_train = loss_train.numpy()
        
        print('Epoch: ', e)
        print('MAE: ', loss_train)
        random.shuffle(train_index)
        lr *= params['LEARNING_RATE_DECAY']
        opt.learning_raye = lr
    
    if params['SAVE_MODEL']:
        model_path = os.path.join(params['SAVED_MODEL_PATH'],'version' +str(model_number))
        tf.keras.models.save_model(model, model_path)
        print('Model is saved at ', model_path)


if __name__ == '__main__':
    #print(os.environ)
    #num_cores = 5
    #tf.config.threading.set_intra_op_parallelism_threads(num_cores)
    #tf.config.threading.set_inter_op_parallelism_threads(num_cores)
    #print(tf.config.get_soft_device_placement())
    for i in range(params['NUM_MODELS']):
        train(i+1+1)