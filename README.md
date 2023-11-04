# Detection of Melt Ponds on Arctic Sea Ice from Infrared Images

This repository develops a segmentation tool that partitions helicopter-borne thermal infrared (TIR) images into sea ice, melt pond, and ocean classes. 
The data used were aquired during the PSP131 ATWAICE campaign [1]. All training images and masks can be investigated in ```preprocess_training.ipynb```.
The model is a U-net with ResNet34 backbone, pretrained on ImageNet. Current work focusses on the integration of visual imagery into the training.

The data used is not published yet.

### Publications
[Link](https://seaice.uni-bremen.de/proceedings-theses-reports/) to Bachelor thesis.

[Link](https://te.ma/art/ut5cb0/reil-melting-ponds-arctic-sea/) to related essay.

![pred_smpl](https://github.com/marlens123/pond_segmentation/assets/80780236/e0298018-ea2d-44a4-9711-a00b69464980)

### Setup
This code requires Python 3.10. Install the required packages using ```pip install -r requirements.txt```.

### Quickstart
If you want to use the current optimized model to segment images and extract melt pond fraction for a specific flight:

- Insert the respective ```netCDF``` file into ```data/prediction/raw/```.
- Run ```python predict.py --data [path_to_netCDF_file]```.
- Predicted images can be found in ```data/prediction/predicted/```. The resulting melt pond fraction can be found in ```metrics/melt_pond_fraction/mpf.csv```.

To fine-tune the model:

- Run ```python fine_tune.py --pref [pref_name] --use_dropout --use_class_weights```, where ```pref_name``` will be used as identifier for the model weights and tracking. By default, the data in ```data/training/``` will be used.
- Evaluation scores can be found in ```metrics/scores/[pref_name].csv```. Resulting model weights can be found in ```weights/best_model[pref_name].h5```

In this work, hyperparameters have been investigated sequentially. To reproduce hyperparameter optimization using k-crossfold validation:

- Run ```sh model_selection.sh```. Model weights will not be stored during hyperparameter tuning.
- This work selected hyperparameter values according to the best iou score averaged accross all folds. For implementation see ```utils/evaluate_hyperparams.py```. Results can be found in ```metrics/hyperparameter_tune_results/[pref_name].csv```.

### Additional Files
This repository covers annotation, preprocessing, training, hyperparameter optimization, and prediction procedures. More information in the respective notebook headers.

```extract_and_annotate.ipynb```: image extraction and preparation for annotation.

```preprocess_training.ipynb```: image and mask preprocessing to reproduce current training data (stored in ```data/training/```).

```prediction_sample.ipynb```: inference examples for different patch size scenarios.

```data/```: data container.

```docs/```: annotation documentation.

```models/```: pre-trained [segmentation models](https://github.com/qubvel/segmentation_models).

```metrics/```: stores model scores and melt pond fraction results.

```utils/```: functions needed for preprocessing, training, prediction.

```weights/```: weights of fine-tuned models.

### Disclaimer
The test data set currently contains only two images and is unlikely to represent the training data distribution, not to mention the distribution of the real data. Therefore, numerical performance estimates should be considered with caution, and should be regarded in combination with qualitative results (```prediction_sample.ipynb```). Hyperparameter optimization was performed using k-crossfold validation to give a better decision base.

### Background
Melt ponds are pools of water on Arctic sea ice that have a strong influence on the Arctic energy budget by increasing the amount of sunlight that is absorbed. 
Accurate quantitative analysis of melt ponds is important for improving Arctic climate model predictions.
Infrared imagery can be used to derive melt pond parameters and thermal properties.

### Model Architecture
<img scr="https://github.com/marlens123/ponds_extended/assets/80780236/84dde17c-6ecd-4608-af7f-7be75de84729" width="200">

![model_architecture|50%](https://github.com/marlens123/ponds_extended/assets/80780236/84dde17c-6ecd-4608-af7f-7be75de84729)

### Disclaimer
The project is the extended version of my Bachelor thesis under the supervision of Dr. Gunnar Spreen ([Remote Sensing Group of Polar Regions](https://seaice.uni-bremen.de/research-group/), University of Bremen)
and Dr. Ulf Krumnack ([Computer Vision Group](https://www.ikw.uni-osnabrueck.de/en/research_groups/computer_vision.html), University of Osnabrück).

### References
[1] Kanzow, Thorsten (2023). The Expedition PS131 of the Research Vessel POLARSTERN to the
Fram Strait in 2022. Ed. by Horst Bornemann and Susan Amir Sawadkuhi. Bremerhaven. DOI: 10.57738/BzPM\_0770\_2023.

### Contact
mareil@uni-osnabrueck.de
