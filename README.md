# Detection of Melt Ponds on Arctic Sea Ice from Infrared Images

Melt ponds are pools of water on Arctic sea ice that have a strong influence on the Arctic energy budget by increasing the amount of sunlight that is absorbed. 
Accurate quantitative analysis of melt ponds is important for improving Arctic climate model predictions.

This repository develops a segmentation tool that partitions helicopter-borne thermal infrared (TIR) images into sea ice, melt pond, and ocean classes. 
The data were aquired during the PSP131 ATWAICE campaign [1].
The model is a U-net with ResNet34 backbone, pretrained on ImageNet. Current work focusses on the integration of visual imagery into the training.

### Model Architecture

<img scr="https://github.com/marlens123/ponds_extended/assets/80780236/84dde17c-6ecd-4608-af7f-7be75de84729" width="200">

![model_architecture|50%](https://github.com/marlens123/ponds_extended/assets/80780236/84dde17c-6ecd-4608-af7f-7be75de84729)

### Current Results

![quali_old](https://github.com/marlens123/pond_segmentation/assets/80780236/7228e021-4630-4367-a4c8-e30b2fdfb3da)
*first column - input flight 9, second column - results flight 9, third column - input flight 16, fourth column - results flight 16. Grey - sea ice, black - melt ponds, white - ocean.*

### Overview
This repository covers annotation, preprocessing, training, hyperparameter optimization, and prediction procedures. 

```extract_and_annotate_sample.ipynb```: image extraction and preparation for annotation.

```preprocessing_sample.ipynb```: image and mask preprocessing.

```training_sample.ipynb```: model training and hyperparameter optimization.

```prediction_sample.ipynb```: inference.

### Setup
This code runs on Python 3.10. For installing the required packages, use ```pip install -r requirements.txt```.
The data used is not published yet.

### Disclaimer
The project is the extended version of my Bachelor thesis under the supervision of Dr. Gunnar Spreen (Polar Remote Sensing Group, University of Bremen)
and Dr. Ulf Krumnack (Computer Vision Group, University of Osnabrück; submission 08/2023).

### References
[1] Kanzow, Thorsten (2023). The Expedition PS131 of the Research Vessel POLARSTERN to the
Fram Strait in 2022. Ed. by Horst Bornemann and Susan Amir Sawadkuhi. Bremerhaven. DOI: 10.57738/BzPM\_0770\_2023.

- pretrained U-net: https://github.com/qubvel/segmentation_models 
- training monitoring: https://www.wandb.com 
- patch prediction: https://github.com/bnsreenu/python_for_microscopists/blob/master/206_sem_segm_large_images_using_unet_with_custom_patch_inference.py
