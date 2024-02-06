# U-net Detection of Melt Ponds on Arctic Sea Ice from Infrared Images

Weights of the model used in the paper can be found at ```weights/base_unet/best_model0302_u4.h5```.

## Setup
This code requires Python 3.10. Install the required packages using ```pip install -r requirements.txt```.

## Quickstart
(Training data and model weights contained in this repository are tracked by Git LFS. To restore them, install Git LFS, run ```git lfs fetch``` and then ```git lfs checkout```).

#### Prediction
If you want to use the current optimized model to segment images and extract melt pond fraction for a specific flight:

- Insert the respective ```netCDF``` file into ```data/prediction/raw/```.
- Run ```python predict.py --pref [pref_name] --data [name_of_netCDF_file]```. To be able to inspect prediction results as grayscale images, add ```--convert_to_grayscale```. If you want to switch model weights, specify ```weights_path [relative_path_to_weights]```. For options see ```weights/```.
- Predicted images can be found in ```data/prediction/predicted/```. The resulting melt pond fraction can be found in ```metrics/melt_pond_fraction/```.

#### Training
To fine-tune the model:

- Run ```python fine_tune.py --pref [pref_name]```, where ```pref_name``` will be used as identifier for the model weights and tracking. By default, the data in ```data/semi_super/``` and hyperparameter setting defined in ```config/semi_01.json``` will be used.
- Evaluation scores can be found in ```metrics/scores/fine_tune/[pref_name].csv```. Resulting model weights can be found in ```weights/```.

#### Hyperparameter Tuning
In this work, hyperparameters have been investigated sequentially. To reproduce hyperparameter optimization using k-crossfold validation:

- Run ```sh model_selection.sh```. Model weights will not be stored during hyperparameter tuning.
- This work selected hyperparameter values according to the best validation iou score averaged over all folds. For implementation see ```utils/evaluate_hyperparams.py```. Results can be found in ```metrics/hyperparameter_tune_results/[pref_name].csv```.
- To test costum hyperparameter values, create a respective config file in ```config/```and run ```python utils/evaluate_hyperparams.py --pref [pref_name] --path_to_config [path_to_custom_config]```.

## Additional Files
This repository covers annotation, preprocessing, training, hyperparameter optimization, and prediction procedures.

```extract_and_annotate.ipynb```: image extraction and preparation for annotation.

```preprocess_training.ipynb```: image and mask preprocessing to reproduce training data.

```prediction_sample.ipynb```: inference examples for different patch size scenarios.

```config/```: Configuration files to set hyperparameters for training.

```data/```: data container.

```docs/```: annotation documentation.

```models/```: pre-trained [segmentation models](https://github.com/qubvel/segmentation_models).

```metrics/```: stores model scores and melt pond fraction results.

```utils/```: functions needed for preprocessing, training, prediction.

```weights/```: weights of fine-tuned models.
