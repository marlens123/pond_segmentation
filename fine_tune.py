import os
import keras
import tensorflow as tf
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from utils.augmentation import get_training_augmentation, get_preprocessing, offline_augmentation
from utils.train_helpers import compute_class_weights, patch_extraction
from utils.data import Dataloder, Dataset
from sklearn.model_selection import KFold
import models.segmentation_models_qubvel as sm
from models.segmentation_models_qubvel.segmentation_models.utils import set_trainable
import argparse
from utils.train import run_train

import wandb
wandb.login()
from wandb.keras import WandbMetricsLogger


parser = argparse.ArgumentParser(description="Model fine-tuning. Default hyperparameter values were optimized during previous experiments.")

# prefix
parser.add_argument("--pref", default="001", type=str, help="Identifier for the run. Model scores will be stored with this prefix.")

# data
parser.add_argument("--X_train", default="data/train_images.npy", type=str, help="Path to training images in .npy file format.")
parser.add_argument("--y_train", default="data/train_masks.npy", type=str, help="Path to training masks in .npy file format.")
parser.add_argument("--X_test", default="data/test_images.npy", type=str, help="Path to testing images in .npy file format.")
parser.add_argument("--y_test", default="data/test_masks.npy", type=str, help="Path to testing masks in .npy file format.")

# hyperparameters
parser.add_argument("--im_size", default=480, type=int, choices=[32, 64, 128, 256, 480], help="Patch size to train on. Choices are constrained because of patch extraction setup.")
parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs. The weights of the best performing training epoch will be stored.")
parser.add_argument("--loss", default="focal_dice", type=str, choices=["categoricalCE", "focal_dice", "focal"], help="Loss function. E.g. 'categorical_CE' or 'focal_dice'. For more options see sm.")
parser.add_argument("--backbone", default="resnet34", type=str, help="U-net backbone to use. For options see sm.")
parser.add_argument("--optimizer", default="Adam", type=str, choices=["Adam", "SGD", "Adamax"], help="Optimizer to use. For options see sm.")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size. Adjust with respect to training set size and patch size.")
parser.add_argument("--augmentation_design", default="on_fly", type=str, choices=[None, "offline", "on_fly"], help="Either None, 'offline' (fixed augmentation before training), or 'on_fly' (while feeding data into the model).")
parser.add_argument("--augmentation_technique", default=4, type=int, choices=[0, 1, 2, 3, 4, 5], help="0 : flip, 1 : rotate, 2 : crop, 3 : brightness contrast, 4 : sharpen blur, 5 : Gaussian noise.")
parser.add_argument("--augmentation_factor", default=2, type=int, help="Magnitude by which the dataset will be increased through augmentation. Only takes effect when augmentation_design is set 'offline'.")
parser.add_argument("--weight_classes", default=True, type=bool, help="If the loss function should account for class imbalance.")
parser.add_argument("--use_dropout", default=True, type=bool, help="If to use dropout layers after upsampling operations in the decoder.")
parser.add_argument("--pretrain", default="imagenet", type=str, choices=["imagenet", None], help="Either 'imagenet' to use encoder weights pretrained on ImageNet or None to train from scratch.")
parser.add_argument("--freeze", default=True, type=bool, help="Only takes effect when pretrain is set to True. Whether to freeze encoder during training or allow fine-tuning of encoder weights.")


def main():
    args = parser.parse_args()
    params = vars(args)

    # load data
    train_images = np.load(params['X_train'])
    train_masks = np.load(params['y_train'])
    test_images = np.load(params['X_test'])
    test_masks = np.load(params['y_test'])

    print(np.unique(train_masks))

    # compute class weights
    if params['weight_classes']:
        class_weights = compute_class_weights(train_masks)
    else:
        class_weights = None

    print("Class weights are...:", class_weights)

    # patch extraction
    X_train, y_train = patch_extraction(train_images, train_masks, size=params['im_size'])
    X_test, y_test = patch_extraction(test_images, test_masks, size=params['im_size'])

    # set augmentation
    on_fly = None
    if params['augmentation_design'] == 'on_fly':
        on_fly = get_training_augmentation(im_size=params['im_size'], mode=params['augmentation_technique'])
    elif params['augmentation_design'] == 'offline':
        X_train, y_train = offline_augmentation(X_train, y_train, im_size=params['im_size'], mode=params['augmentation_technique'], factor=params['augmentation_factor'])

    # construct model
    model = sm.Unet(params['backbone'], input_shape=(params['im_size'], params['im_size'], 3), classes=3, activation='softmax', encoder_weights=params['pretrain'],
                    decoder_use_dropout=params['use_dropout'], encoder_freeze=params['freeze'])  

    print(model.summary())

    run = wandb.init(project='pond_segmentation',
                        group=params['pref'],
                        name=params['pref'],
                        config={
                        "loss_function": params['loss'],
                        "batch_size": params['batch_size'],
                        "backbone": params['backbone'],
                        "optimizer": params['optimizer'],
                        "train_transfer": params['pretrain'],
                        "augmentation": params['augmentation_design']
                        }
    )
    config = wandb.config

    # run training
    _, _ = run_train(pref=params['pref'], X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_epochs=params['num_epochs'],
                  loss=params['loss'], backbone=params['backbone'], optimizer=params['optimizer'], batch_size=params['batch_size'], 
                  model=model, augmentation=on_fly, class_weights=class_weights)

    wandb.join()

if __name__ == "__main__":
    main()

    



