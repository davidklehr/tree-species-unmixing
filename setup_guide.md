## Setup Workspace
### Installation

1. Clone this repository \
    ``` git clone https://github.com/davidklehr/tree-species-unmixing.git ```
    

2. Install python and anaconda(not necessary, but recommended)

I have managed my python version and the used python packages in <a href="./requirements.txt" >requirements.txt</a> using anaconda.
I would recommend this, but of course you are free to manage the used package versions yourself.

### Python environment

First create a new conda environment if you use anaconda.\
    ```conda create --name Synth_Mix_ANN python=3.9 ``` \
    ``` conda activate Synth_Mix_ANN ```


Afterwards install the required packages in your activated environment. \
    ``` conda install numpy ``` \
    ``` conda install gdal ``` \
    ``` conda install rasterio ``` \
    ``` conda install tqdm ``` \
    ``` conda install tensorflow=2.10.0 ``` \
    ```conda install anaconda::joblib``` \

Now everything should be settled up. 

### Folder Structure
During my processing, I used the following fixed folder structure. I recommend you a similar structure.
```
/Tree_Species_Unmixing
   ├-/SplineReconstruction
   │  │ # This Folder contains the FORCE Datacube in its tile structure.
   |  ├ X0001_Y0001
   |  ├ X0001_Y0002
   |  ├ ...
   |  ├ X0002_Y0001
   |  ├ X0002_Y0002
   │  └-...
   ├-/unmixing
   │  ├-/1_rasterized_points
   |  |  └─ ...
   │  ├-/2_pure_data
   |  |  └─ ...
   │  ├-/3_mixed_data
   |  |  └─ ...
   │  ├-/4_trained_model
   |  |  └─ ...
   │  ├-/5_prediction
   |  |  └─ ...
   │  ├-/6_prediction_normalized
   |  |  └─ ...
   |  └─ ...
   └-/python
      ├-params.py
      ├-1_rasterize_pure_points.py
      ├-2_extract_spectral.py
      ├-3_synthetic_mixing.py
      ├-4_train_NN.py
      ├-5_mapping.py
      ├-normalization_params.py
      ├-normalize_fractions_parallel.py
      └-...
```

The first folder ("SplineReconstruction") contains the feature raster, which is the reconstructed S2 time-series. In principle it can contain any kind of features you want to use for classification. As the reconstruction is performed using FORCE (Link) it is stored in a datacube structure.
Anyway, the workflow will also work, if the feature raster image is not stored in this structure but in a single .tif raster.
However, I strongly recommend this structure, because otherwise parallellization in later scripts will have no impact and the processing time will increase.

The second folder ("unmixing") contains subfolders, where intermediate results of the individual python scripts will be stored.

The last folder only contains the relevant python scripts and the paramterfile(s), for easy comannd line commands.

## Processing

Before starting the processing steps you should have your time series data ready, meaning one raster (or more, if you use more bands; e.g. one raster for each band) where each cell contains the spectral time series data.
If you used FORCE (TSA-module) for preprocessing your satellite data, they are stored correctly already and in datacube structure.
Otherwise it might be necessary to adapt the scripts a little.

**Step 1** - Building your spectral library \
\
    The first step actually start with the necessary information of locations of pure tree species pixel. It is of utmost importance to use only high-quality pixels with no clue of intermixing with other species.
    You can store the data in a vectordata format of you choise (recommended .shp or .gpkg) with one column referring to the tree species (e.g. 1 = beech; 2 = oak; ...)
    \
    Adapt all folderfiles and parameters in the parameter file (params.py - lines 1 to 56 for script one, two, and three). Afterwards you can run script one to three.
    ``` python *folder_of_processing*/Tree_Species_Unmixing/python/1_avg_rasterize_pure_points.py``` \ 
    ``` python *folder_of_processing*/Tree_Species_Unmixing/python/2_avg_extract_spectral.py ``` \
    ``` python *folder_of_processing*/Tree_Species_Unmixing/python/3_avg_synthetic_mixing.py ``` \
\
    With this three skripts you extract the spectral time series information at the points of your selected pure tree species pixels (script one and two).
    Afterwards you start a randomized synthatical mixing of the pure endmembers (in relation complexity of your choice) and store this new artifical time series and the according pure species fractions (in the created folder '3_mixed_data').
    If you perform an ensamble approach, you will create an individual library for every model you will train in the next step.

**Step 2** - Train the Neural Network \
\
In the second step you will train your neural network (or multiple networks if you are using the ensamble approach). This isthe most time-consuming step, depending on your hardware and chosen number of epochs.
The individual networks are stored in the created folder 4_trained_model.
Make sure to adapt your setting and all folder paths in the parameter file.

```bash
    python *folder_of_processing*/Tree_Species_Unmixing/python/4_train_mulit_ANN.py
```

**Step 3** - Apply the model for tree species fraction prediction \
\
In the last step you will apply the trained models to your original time-series raster data. Here the power of data-cubing comes into play, as you can set the number processed tiles in parallel depending on your hardware.

```bash
    python *folder_of_processing*/Tree_Species_Unmixing/python/5_mapping_multimodel_parallel.py
```

After performing this last of the five python scripts, your fraction raster is ready for inspection and/or validation.
It might be reasonable to normalize the predicted raster, that the sum of all fractions equals to 0. Depending on the model, the time series, and differentiablity of the tree species this might not always be the case.