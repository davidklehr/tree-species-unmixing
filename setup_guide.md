## Setup Workspace
### Installation

1. Clone this repository
    ```bash
    git clone https://github.com/davidklehr/tree-species-unmixing.git
    ```

2. Install python and anaconda(not necessary, but recommended)

I have managed my python version and the used python packages in requirements.txt using anaconda.
<a href="./requirements.txt" >Test</a>
I would recommend this, but of course you are free to manage the used package versions yourself.

### Python environment

3. Install python packages:
First create a new conda environment if you use anaconda.
    ```bash
    conda create --name Synth_Mix_ANN python=3.9
    conda activate Synth_Mix_ANN
    ```

Afterwards install the required packages in your activated environment
    ```bash
    conda install numpy
    conda install gdal
    conda install rasterio
    conda install tqdm
    conda install tensorflow=2.10.0
    conda install anaconda::joblib
    
    ```
Now everything should be settled up. 

### Folder Structure
During my processing, I used a fixed folder structure. I recommend you a similar structure.

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
   ├-/Unmixing
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
The first folder...
Anyway, the workflow will also work, if the Time-Series Image is not stored in datacube structure but in a single .tif raster.
However, I strongly recommend this structure, because otherwise parallellization in later scripts will have no impact and the processing time will increase.

## Processing

1. Schritt 1 ausführen:
    ```bash
    python src/step_1.py --input "input_datei.csv"
    ```
2. Schritt 2:
    ```bash
    python src/step_2.py --input "output_schritt_1.csv"
    ```
3. ... (alle Schritte erklären)

### Verarbeitungsschritte
- **Schritt 1**: [Beschreibung]
- **Schritt 2**: [Beschreibung]
- **Schritt 3**: [Beschreibung]
- **Schritt 4**: [Beschreibung]
- **Schritt 5**: [Beschreibung]

### Beispiel
```bash
python src/step_1.py --input "input_datei.csv"
python src/step_2.py --input "output_schritt_1.csv"
...
