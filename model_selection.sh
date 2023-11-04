#!/bin/bash

# Script to reproduce the model selection process. 
# Runs model training with different hyperparameter values and stores results 
# in 'metrics/hyperparameter_tune/' folder.
# K-crossfold validation is used because of small data.
# For more information about the hyperparameters and selection process,
# see the 'utils/evaluate_hyperparams.py' function and the Bachelor thesis report.
# A smaller dataset was used to obtain the results in the thesis.

echo "Investigating patch size..."

python utils/evaluate_hyperparams.py --pref patch_size_32 --im_size 32 --loss categoricalCE --augmentation_design None --batch_size 32
python utils/evaluate_hyperparams.py --pref patch_size_64 --im_size 64 --loss categoricalCE --augmentation_design None --batch_size 16
python utils/evaluate_hyperparams.py --pref patch_size_128 --im_size 128 --loss categoricalCE --augmentation_design None --batch_size 8
python utils/evaluate_hyperparams.py --pref patch_size_256 --im_size 256 --loss categoricalCE --augmentation_design None --batch_size 4
python utils/evaluate_hyperparams.py --pref patch_size_480 --im_size 480 --loss categoricalCE --augmentation_design None --batch_size 2

echo "Investigating loss function..."

python utils/evaluate_hyperparams.py --pref loss_focal --im_size 480 --loss focal --use_class_weights --augmentation_design None --batch_size 2
python utils/evaluate_hyperparams.py --pref loss_focal_dice --im_size 480 --loss focal_dice --use_class_weights --augmentation_design None --batch_size 2

echo "Investigating dropout..."

python utils/evaluate_hyperparams.py --pref use_dropout --im_size 480 --loss focal_dice --use_class_weights --use_dropout --augmentation_design None --batch_size 2

echo "Investigating pretraining strategy..."

python utils/evaluate_hyperparams.py --pref freeze_encoder --im_size 480 --loss focal_dice --use_class_weights --use_droppout --augmentation_design None --batch_size 2
python utils/evaluate_hyperparams.py --pref train_from_scratch --im_size 480 --loss focal_dice --use_class_weights --use_dropout --augmentation_design None --batch_size 2

echo "Investigating augmentation technique..."

python utils/evaluate_hyperparams.py --pref flip --im_size 480 --loss focal_dice --use_class_weights --use_dropout --augmentation_technique 0 --batch_size 2
python utils/evaluate_hyperparams.py --pref rotate --im_size 480 --loss focal_dice --use_class_weights --use_dropout --augmentation_technique 1 --batch_size 2
python utils/evaluate_hyperparams.py --pref crop --im_size 480 --loss focal_dice --use_class_weights --use_dropout --augmentation_technique 2 --batch_size 2
python utils/evaluate_hyperparams.py --pref brightness_contrast --im_size 480 --loss focal_dice --use_class_weights --use_dropout --augmentation_technique 3 --batch_size 2
python utils/evaluate_hyperparams.py --pref Gauss_noise --im_size 480 --loss focal_dice --use_class_weights --use_dropout --augmentation_technique 5 --batch_size 2
python utils/evaluate_hyperparams.py --pref sharpen_blur --im_size 480 --loss focal_dice --use_class_weights --use_dropout --augmentation_technique 4 --batch_size 2

echo "Investigating augmentation design..."

python utils/evaluate_hyperparams.py --pref on_fly --im_size 480 --loss focal_dice --use_class_weights --augmentation_design on_fly --batch_size 2
python utils/evaluate_hyperparams.py --pref offline --im_size 480 --loss focal_dice --use_class_weights --augmentation_design offline --batch_size 2

echo "Model selection done. Logs are stored in 'metrics/hyperparameter_tune/'."