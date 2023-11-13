import numpy as np
from utils.augmentation import get_training_augmentation, offline_augmentation
from utils.train_helpers import compute_class_weights, patch_extraction
import models.segmentation_models_qubvel as sm
import argparse
from utils.train import run_train
import json

import wandb

parser = argparse.ArgumentParser(description="Model fine-tuning. Default hyperparameter values were optimized during previous experiments.")

# prefix
parser.add_argument("--pref", default="ft_001", type=str, help="Identifier for the run. Model scores will be stored with this prefix.")

# data
parser.add_argument("--path_to_X_train", default="data/training/train_images.npy", type=str, help="Path to training images in .npy file format.")
parser.add_argument("--path_to_y_train", default="data/training/train_masks.npy", type=str, help="Path to training masks in .npy file format.")
parser.add_argument("--path_to_X_test", default="data/training/test_images.npy", type=str, help="Path to testing images in .npy file format.")
parser.add_argument("--path_to_y_test", default="data/training/test_masks.npy", type=str, help="Path to testing masks in .npy file format.")

# hyperparameters
parser.add_argument("--path_to_config", default="config/best_unet.json", type=str, help="Path to config file that stores hyperparameter setting. For more information see 'config/README.md'.")

parser.add_argument("--use_wandb", action='store_true', help="Whether to use wandb for train monitoring.")



def main():
    args = parser.parse_args()
    params = vars(args)

    with open(params['path_to_config']) as f:
        cfg = json.load(f)

    cfg_model = cfg['model']
    cfg_augmentation = cfg['augmentation']
    cfg_training = cfg['training']

    wandb = params['use_wandb']

    # load data
    train_images = np.load(params['path_to_X_train'])
    train_masks = np.load(params['path_to_y_train'])
    test_images = np.load(params['path_to_X_test'])
    test_masks = np.load(params['path_to_y_test'])

    print(np.unique(train_masks))

    # compute class weights
    if cfg_training['use_class_weights']:
        class_weights = compute_class_weights(train_masks)
    else:
        class_weights = None

    print("Class weights are...:", class_weights)

    # patch extraction
    X_train, y_train = patch_extraction(train_images, train_masks, size=cfg_model['im_size'])
    X_test, y_test = patch_extraction(test_images, test_masks, size=cfg_model['im_size'])

    # set augmentation
    on_fly = None
    if cfg_augmentation['design'] == 'on_fly':
        on_fly = get_training_augmentation(im_size=cfg_model['im_size'], mode=cfg_augmentation['technique'])
    elif cfg_augmentation['design'] == 'offline':
        X_train, y_train = offline_augmentation(X_train, y_train, im_size=cfg_model['im_size'], mode=cfg_augmentation['technique'], factor=params['factor'])

    # set pretraining
    if cfg_model['pretrain'] == "none":
        cfg_model['pretrain'] = None

    # construct model
    if cfg_model['architecture'] == 'base_unet':
        model = sm.Unet(cfg_model['backbone'], input_shape=(cfg_model['im_size'], cfg_model['im_size'], 3), classes=cfg_model['classes'], activation=cfg_model['activation'], encoder_weights=cfg_model['pretrain'],
                        decoder_use_dropout=cfg_model['use_dropout'], encoder_freeze=cfg_model['freeze'])  
   
    elif cfg_model['architecture'] == 'att_unet':
        model = sm.Unet(cfg_model['backbone'], input_shape=(cfg_model['im_size'], cfg_model['im_size'], 3), classes=cfg_model['classes'], activation=cfg_model['activation'], encoder_weights=cfg_model['pretrain'],
                        decoder_use_dropout=cfg_model['use_dropout'], encoder_freeze=cfg_model['freeze'], decoder_add_attention=True)  

    print(model.summary())

    if wandb:
        wandb.login()
        run = wandb.init(project='pond_segmentation',
                            group=params['pref'],
                            name=params['pref'],
                            config={
                            "loss_function": cfg_training['loss'],
                            "batch_size": cfg_training['batch_size'],
                            "backbone": cfg_training['backbone'],
                            "optimizer": cfg_training['optimizer'],
                            "train_transfer": cfg_model['pretrain'],
                            "augmentation": cfg_augmentation['design']
                            }
        )
        config = wandb.config

    # run training
    _, _ = run_train(pref=params['pref'], X_train_ir=X_train, y_train=y_train, X_test_ir=X_test, y_test=y_test, train_config=cfg_training,
            model=model, use_wandb=wandb, augmentation=on_fly, class_weights=class_weights)

    if wandb:
        wandb.join()

if __name__ == "__main__":
    main()

    



