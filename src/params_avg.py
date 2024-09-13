params = {
    # ------------ Script 1 -------------
    # (*) indicates MUST change !!
    
    # (*) Tree endmember points (.shp or .gpkg)
    'PURE_POINTS_PATH': '*path_to_your_vectordata*'
    
    # Column contains tree id
    # each tree species and the background class needs one individual number. E.g. Beech = 1; Oak = 2; ...
    'TREE_CLASS_COLUM_NAME': 'spec',
    
    # (*) Output directory for rasterized points
    'RASTERIZED_POINT_DIR': '*/Tree_Species_Unmixing/Unmixing/1_rasterized_points',
    
    # (*) Tile data directory.
    # This is used for extracting the extent of the tile, as well as getting data in other Scripts
    'DATA_CUBE_DIR': '*/Tree_Species_Unmixing/SplineReconstruction',
    
    # Dimension of the FORCE - tile ( [3000x3000] for 10 m resolution or [1000x1000] for 30 m resolution)
    'RASTER_PIXEL_NUM': 3000,
    
    # Default nodata output, this is used for both rasterized points and later mapping results
    'NO_DATA_OUTPUT': 255,
    
    
    # ------------ Script 2 -------------
    # List of all years used for training and mapping, put the numbers in STRING!
    'YEAR_LIST': ['2022'],
    
    # Bands used
    'BAND_LIST': ['BLU', 'GRN', 'RED', 'RE1', 'RE2', 'RE3', 'NIR', 'BNR', 'SW1', 'SW2', 'NDV', 'EVI'],
    
    # All tree IDs
    'TREE_CLASS_LIST': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    
    # (*) Output directory for extracting the pure spectra
    'EXTRACTED_SPECTRA_DIR': '*/Tree_Species_Unmixing/Unmixing/2_pure_data',
    
    # ------------ Script 3 -------------
    # Number of mixed spectral data
    'MIXED_NUM': 256000,
    
    # List of mixing number, how many classes can be mixed in one mixture.
    # This should aim to represent the situatation in field, meaning in this example, a mixture of up to three classes are possible
    'MIXTURE_LIST': [1, 2, 3],
    
    # Weights of mixing number, defaut all are equally drafted. Increase number to increase the chance to draft
    # For example [1, 5, 5] will increase more chances to have 2-class and 3-class mixtures
    'MIXTURE_WEIGHTS': [1, 5, 5],
    
    # (*) Output directory for mixed spectra spectra
    'MIXED_SPECTRA_DIR': '*/Tree_Species_Unmixing/Unmixing/3_mixed_data',
    
    # Number of models in the ensamble approach
    # For each model an individual synthetic mixture library is calculated
    'NUM_MODELS': 10,
    
    # ------------ Script 4 -------------
    # Numbers of Neural Network (NN) hidden layers
    'NN_hidden_layer_num' : 5,
    
    # Numbers of nodes in each hidden layer
    'NN_hidden_layer_nodes' : 128,
    
    # Normalize input data, i.e., scalling the input to [0, 1] to train faster.
    'NORMALIZE_INPUT': True,
    
    # Batch size for each iteration
    'TRAINING_BATCH_SIZE': 256,
    
    # Initial learning rate 
    'LEARNING_RATE': 1e-3,
    
    # Proportion of learning rate decrease after each epoch, 0 < value <= 1
    'LEARNING_RATE_DECAY': 0.5,
    
    # How many time to train the model with the entire dataset.
    'EPOCHS': 250,
    
    # Save model after finished training
    'SAVE_MODEL': True,
    
    # (*) Location of saved model
    'SAVED_MODEL_PATH': '*/Tree_Species_Unmixing/Unmixing/4_trained_model',
    
    # ------------ Script 5 -------------
    # (*) Directory for mapping result
    'PREDICTION_DIR': '*/Tree_Species_Unmixing/Unmixing/5_prediction',
    
    # Tree names for fraction output, each per band  
    'TREE_NAME_LIST': ['Beech', 'Oak', 'Maple', 'Alder', 'Spruce', 'Douglas Fir', 'Pine', 'Larch', 'Fir', 'Ground']
}