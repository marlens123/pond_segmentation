# Detection of Melt Ponds on Arctic Sea Ice from Infrared Images

This repository develops a segmentation tool that partitions helicopter-borne thermal infrared (TIR) images into sea ice, melt pond, and ocean classes. 
The data used were aquired during the PSP131 ATWAICE campaign [1]. Labeled training images can be investigated in ```preprocess_training.ipynb```.
The current approach uses a pre-trained Attention-U-net with ResNet34 backbone. The architecture can be changed to a basic U-net or PSP-net. Current work focusses on enhancing the dataset (```experimental/semi_supervised/```) and fusion of TIR with corresponding VIS images (```experimental/vis_ir_fusion/```).

The data used is not published yet.

```data/training/```contains the original manually labeled dataset (16 training images). ```data/semi_super/```contains the dataset extended by pseudo-labeled model predictions (204 training images).

## Publications
[Link](https://seaice.uni-bremen.de/proceedings-theses-reports/) to Bachelor thesis.

[Link](https://te.ma/art/ut5cb0/reil-melting-ponds-arctic-sea/) to related essay.

![pred_smpl](https://github.com/marlens123/pond_segmentation/assets/80780236/e0298018-ea2d-44a4-9711-a00b69464980)

## Table of Contents
1. [Setup](https://github.com/marlens123/pond_segmentation/blob/main/README.md#setup)
2. [Quickstart](https://github.com/marlens123/pond_segmentation/blob/main/README.md#quickstart)
   1. [Prediction](https://github.com/marlens123/pond_segmentation/blob/main/README.md#prediction)
   2. [Training](https://github.com/marlens123/pond_segmentation/blob/main/README.md#training)
   3. [Hyperparameter Tuning](https://github.com/marlens123/pond_segmentation/blob/main/README.md#hyperparameter-tuning)
3. [Additional Files](https://github.com/marlens123/pond_segmentation/blob/main/README.md#additional-files)
4. [Interpretation of Results](https://github.com/marlens123/pond_segmentation/blob/main/README.md#interpretation-of-results)
5. [Background](https://github.com/marlens123/pond_segmentation/blob/main/README.md#background)
6. [Model Architecture](https://github.com/marlens123/pond_segmentation/blob/main/README.md#model-architecture)
9. [Disclaimer](https://github.com/marlens123/pond_segmentation/blob/main/README.md#disclaimer)
8. [References](https://github.com/marlens123/pond_segmentation/blob/main/README.md#references)

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
- Evaluation scores can be found in ```metrics/scores/[pref_name].csv```. Resulting model weights can be found in ```weights/best_model[pref_name].h5```.

#### Hyperparameter Tuning
In this work, hyperparameters have been investigated sequentially. To reproduce hyperparameter optimization using k-crossfold validation:

- Run ```sh model_selection.sh```. Model weights will not be stored during hyperparameter tuning.
- This work selected hyperparameter values according to the best validation iou score averaged over all folds. For implementation see ```utils/evaluate_hyperparams.py```. Results can be found in ```metrics/hyperparameter_tune_results/[pref_name].csv```.
- To test costum hyperparameter values, create a respective config file in ```config/```and run ```python utils/evaluate_hyperparams.py --pref [pref_name] --path_to_config [path_to_custom_config]```.

## Additional Files
This repository covers annotation, preprocessing, training, hyperparameter optimization, and prediction procedures.

```extract_and_annotate.ipynb```: image extraction and preparation for annotation.

```preprocess_training.ipynb```: image and mask preprocessing to reproduce current training data (stored in ```data/training/flight9_flight16/```).

```prediction_sample.ipynb```: inference examples for different patch size scenarios.

```config/```: Configuration files to set hyperparameters for training.

```data/```: data container.

```docs/```: annotation documentation.

```models/```: pre-trained [segmentation models](https://github.com/qubvel/segmentation_models).

```metrics/```: stores model scores and melt pond fraction results.

```utils/```: functions needed for preprocessing, training, prediction.

```weights/```: weights of fine-tuned models.

## Interpretation of Results
The test data set currently contains only two images and is unlikely to represent the training data distribution, not to mention the distribution of the real data. Therefore, numerical performance estimates should be considered with caution, and should be regarded in combination with qualitative results (```prediction_sample.ipynb```). Hyperparameter optimization was performed using k-crossfold validation to give a better decision base.

## Background
Melt ponds are pools of water on Arctic sea ice that have a strong influence on the Arctic energy budget by increasing the amount of sunlight that is absorbed. 
Accurate quantitative analysis of melt ponds is important for improving Arctic climate model predictions.
Infrared imagery can be used to derive melt pond parameters and thermal properties.

## Model Architecture
<img scr="https://github.com/marlens123/ponds_extended/assets/80780236/84dde17c-6ecd-4608-af7f-7be75de84729" width="200">

![architecture_att](https://github.com/marlens123/pond_segmentation/assets/80780236/f7c59002-b034-4fc0-b010-dec8068261d6)

<img src="[https://i.imgur.com/ZWnhY9T.png](https://github.com/marlens123/pond_segmentation/assets/80780236/f7c59002-b034-4fc0-b010-dec8068261d6)" width=50% height=50%>


## Disclaimer
The project is the extended version of my Bachelor thesis under the supervision of Dr. Gunnar Spreen ([Remote Sensing Group of Polar Regions](https://seaice.uni-bremen.de/research-group/), University of Bremen)
and Dr. Ulf Krumnack ([Computer Vision Group](https://www.ikw.uni-osnabrueck.de/en/research_groups/computer_vision.html), University of Osnabrück).

## References
[1] Kanzow, Thorsten (2023). The Expedition PS131 of the Research Vessel POLARSTERN to the
Fram Strait in 2022. Ed. by Horst Bornemann and Susan Amir Sawadkuhi. Bremerhaven. DOI: 10.57738/BzPM\_0770\_2023.

**Contact**: mareil@uni-osnabrueck.de
