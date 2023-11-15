"""
NOT UP-TO-DATE!

"""


#### same process as fine_tune.py but with early fusion of vis and ir images
#### expects vis and ir images to be aligned and in the same shape
#### disabled offline augmentation for now

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
from utils.augmentation import get_training_augmentation, offline_augmentation
from utils.train_helpers import compute_class_weights, patch_extraction
import models.segmentation_models_qubvel as sm
import argparse
from utils.train import run_train

import wandb
wandb.login()


parser = argparse.ArgumentParser(description="Model fine-tuning. Default hyperparameter values were optimized during previous experiments.")

# prefix
parser.add_argument("--pref", default="ft_001", type=str, help="Identifier for the run. Model scores will be stored with this prefix.")

# data
parser.add_argument("--X_train_ir", default="data/training/train_images.npy", type=str, help="Path to training images in .npy file format.")
parser.add_argument("--y_train", default="data/training/train_masks.npy", type=str, help="Path to training masks in .npy file format.")
parser.add_argument("--X_train_vis", default="data/training/train_images.npy", type=str, help="Path to training images in .npy file format.")
parser.add_argument("--X_test_ir", default="data/training/test_images.npy", type=str, help="Path to testing images in .npy file format.")
parser.add_argument("--y_test", default="data/training/test_masks.npy", type=str, help="Path to testing masks in .npy file format.")
parser.add_argument("--X_test_vis", default="data/training/test_images.npy", type=str, help="Path to testing images in .npy file format.")

# hyperparameters
parser.add_argument("--im_size", default=480, type=int, choices=[32, 64, 128, 256, 480], help="Patch size to train on. Choices are constrained because of patch extraction setup.")
parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs. The weights of the best performing training epoch will be stored.")
parser.add_argument("--loss", default="focal_dice", type=str, choices=["categoricalCE", "focal_dice", "focal"], help="Loss function. E.g. 'categorical_CE' or 'focal_dice'. For more options see sm.")
parser.add_argument("--backbone", default="resnet34", type=str, help="U-net backbone to use. For options see sm.")
parser.add_argument("--optimizer", default="Adam", type=str, choices=["Adam", "SGD", "Adamax"], help="Optimizer to use. For options see sm.")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size. Adjust with respect to training set size and patch size.")
parser.add_argument("--augmentation_design", default="on_fly", type=str, choices=["none", "on_fly"], help="Either None or 'on_fly' (while feeding data into the model).")
parser.add_argument("--augmentation_technique", default=4, type=int, choices=[0, 1, 2, 3, 4, 5], help="0 : flip, 1 : rotate, 2 : crop, 3 : brightness contrast, 4 : sharpen blur, 5 : Gaussian noise.")
parser.add_argument("--use_class_weights", action='store_true', help="If the loss function should account for class imbalance.")
parser.add_argument("--use_dropout", action='store_true', help="If to use dropout layers after upsampling operations in the decoder.")
parser.add_argument("--pretrain", default="imagenet", type=str, choices=["imagenet", "none"], help="Either 'imagenet' to use encoder weights pretrained on ImageNet or None to train from scratch.")
parser.add_argument("--freeze", action='store_true', help="Only takes effect when pretrain is not None. Whether to freeze encoder during training or allow fine-tuning of encoder weights.")


def main():
    args = parser.parse_args()
    params = vars(args)

    ### CHANGED ###

    # load data
    train_images_ir = np.load(params['X_train_ir'])
    train_masks = np.load(params['y_train'])
    test_images_ir = np.load(params['X_test_ir'])
    test_masks = np.load(params['y_test'])

    train_images_vis = np.load(params['X_train_vis'])
    test_images_vis = np.load(params['X_test_vis'])


    # compute class weights from ir imagery
    if params['use_class_weights']:
        class_weights = compute_class_weights(train_masks)
    else:
        class_weights = None

    print("Class weights are...:", class_weights)

    ### CHANGED ### (removed patch extraction)

    X_train_ir = train_images_ir
    y_train = train_masks
    X_train_vis = train_images_vis
    X_test_ir = test_images_ir
    X_test_vis = test_images_vis
    y_test = test_masks

    # set augmentation
    on_fly = None
    if params['augmentation_design'] == 'on_fly':
        on_fly = get_training_augmentation(im_size=params['im_size'], mode=params['augmentation_technique'])

    # set pretraining
    if params['pretrain'] == "none":
        params['pretrain'] = None


    ### CHANGED ###

    # construct model with 4 channels bcs of concatenated modalities
    model = sm.Unet(params['backbone'], input_shape=(params['im_size'], params['im_size'], 4), classes=3, activation='softmax', encoder_weights=params['pretrain'],
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
    _, _ = run_train(pref=params['pref'], X_train_ir=X_train_ir, y_train=y_train, X_test_ir=X_test_ir, y_test=y_test, X_train_vis=X_train_vis, X_test_vis=X_test_vis, num_epochs=params['num_epochs'],
            loss=params['loss'], backbone=params['backbone'], optimizer=params['optimizer'], batch_size=params['batch_size'], 
            model=model, augmentation=on_fly, class_weights=class_weights)

    wandb.join()

if __name__ == "__main__":
    main()