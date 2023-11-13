#!/bin/bash

# Script to reproduce the model selection process. 
# Runs model training with different hyperparameter values and stores results 
# in 'metrics/hyperparameter_tune/' folder.
# K-crossfold validation is used because of small data.
# For more information about the hyperparameters and selection process,
# see the 'utils/evaluate_hyperparams.py' function and the Bachelor thesis report.
# A smaller dataset was used to obtain the results in the thesis.

echo "Investigating patch size..."

python utils/evaluate_hyperparams.py --pref patchsize_32 --path_to_config config/hyperparameter_tune/patchsize_32.json
python utils/evaluate_hyperparams.py --pref patchsize_64 --path_to_config config/hyperparameter_tune/patchsize_64.json
python utils/evaluate_hyperparams.py --pref patchsize_128 --path_to_config config/hyperparameter_tune/patchsize_128.json
python utils/evaluate_hyperparams.py --pref patchsize_256 --path_to_config config/hyperparameter_tune/patchsize_256.json
python utils/evaluate_hyperparams.py --pref patchsize_480 --path_to_config config/hyperparameter_tune/patchsize_480.json

echo "Investigating loss function..."

python utils/evaluate_hyperparams.py --pref loss_focal ---path_to_config config/hyperparameter_tune/loss_focal.json
python utils/evaluate_hyperparams.py --pref loss_focal_dice --path_to_config config/hyperparameter_tune/loss_focal_dice.json

echo "Investigating dropout..."

python utils/evaluate_hyperparams.py --pref use_dropout --path_to_config config/hyperparameter_tune/use_dropout.json

echo "Investigating pretraining strategy..."

python utils/evaluate_hyperparams.py --pref freeze_encoder --path_to_config config/hyperparameter_tune/freeze_encoder.json
python utils/evaluate_hyperparams.py --pref train_from_scratch --path_to_config config/hyperparameter_tune/train_from_scratch.json

echo "Investigating augmentation technique..."

python utils/evaluate_hyperparams.py --pref augment_flip --path_to_config config/hyperparameter_tune/augment_flip.json
python utils/evaluate_hyperparams.py --pref augment_rotate --path_to_config config/hyperparameter_tune/augment_rotate.json
python utils/evaluate_hyperparams.py --pref augment_crop --path_to_config config/hyperparameter_tune/augment_crop.json
python utils/evaluate_hyperparams.py --pref augment_brightness_contrast --path_to_config config/hyperparameter_tune/augment_brightness_contrast.json
python utils/evaluate_hyperparams.py --pref augment_Gauss --path_to_config config/hyperparameter_tune/augment_Gauss.json
python utils/evaluate_hyperparams.py --pref augment_sharpen_blur --path_to_config config/hyperparameter_tune/augment_sharpen_blur.json

echo "Investigating augmentation design..."

python utils/evaluate_hyperparams.py --pref augment_onfly --path_to_config config/hyperparameter_tune/augment_onfly.json
python utils/evaluate_hyperparams.py --pref augment_offline --path_to_config config/hyperparameter_tune/augment_offline.json

echo "Model selection done. Logs are stored in 'metrics/hyperparameter_tune/'."