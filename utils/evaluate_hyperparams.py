import sys
import os

# add parent directory to system path to be able to assess functions from root
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import argparse
import csv
from sklearn.model_selection import KFold
import models.segmentation_models_qubvel as sm
from utils.augmentation import get_training_augmentation, offline_augmentation
from utils.train_helpers import compute_class_weights, patch_extraction
from utils.train import run_train
import json

import wandb

parser = argparse.ArgumentParser(description="Model fine-tuning. Default hyperparameter values were optimized during previous experiments.")

# prefix
parser.add_argument("--pref", default="ho_001", type=str, help="Identifier for the run. Model scores will be stored with this prefix.")

# data
parser.add_argument("--X", default="data/training/train_images.npy", type=str, help="Path to training images in .npy file format.")
parser.add_argument("--y", default="data/training/train_masks.npy", type=str, help="Path to training masks in .npy file format.")

# hyperparameters
parser.add_argument("--path_to_config", default="config/hyperparameter_tune/patchsize_32.json", type=str, help="Path to config file that stores hyperparameter setting. For more information see 'config/README.md'.")

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
    X = np.load(params['X'])
    y = np.load(params['y'])

    # set augmentation
    on_fly = None
    if cfg_augmentation['design'] == 'on_fly':
        on_fly = get_training_augmentation(im_size=cfg_model['im_size'], mode=cfg_augmentation['technique'])

    # set pretraining
    if cfg_model['pretrain'] == "none":
        cfg_model['pretrain'] = None

    # construct model
    model = sm.Unet(cfg_model['backbone'], input_shape=(cfg_model['im_size'], cfg_model['im_size'], 3), classes=cfg_model['classes'], activation=cfg_model['activation'], encoder_weights=cfg_model['pretrain'],
                    decoder_use_dropout=cfg_model['use_dropout'], encoder_freeze=cfg_model['freeze'])  

    print(model.summary())

    # crossfold setup
    num_folds = 4

    val_loss_per_fold = []
    val_iou_per_fold = []           

    # define crossfold validator with random split
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=14)

    fold_no = 1
    base_pref = params['pref']

    for train, test in kfold.split(X, y):

        # add fold number to prefix
        pref = base_pref + "_foldn{}".format(fold_no)

        # compute class weights
        if cfg_training['use_class_weights']:
            class_weights = compute_class_weights(y[train])
            print("Class weights are...:", class_weights)
        else:
            class_weights = None

        # patch extraction
        X_train, y_train = patch_extraction(X[train], y[train], size=cfg_model['im_size'])
        X_test, y_test = patch_extraction(X[test], y[test], size=cfg_model['im_size'])

        # offline augmentation if selected
        if cfg_augmentation['design'] == 'offline':
            X_train, y_train = offline_augmentation(X_train, y_train, im_size=cfg_model['im_size'], mode=cfg_augmentation['technique'], factor=cfg_augmentation['factor'])

        if wandb:
            wandb.login()
            # tracking configuration
            run = wandb.init(project='pond_segmentation',
                                group=params['pref'],
                                name='foldn_{}'.format(fold_no),
                                config={
                                "loss_function": cfg_training['loss'],
                                "batch_size": cfg_training['batch_size'],
                                "backbone": cfg_training['backbone'],
                                "optimizer": cfg_training['optimizer'],
                                "train_transfer": cfg_model['pretrain'],
                                "augmentation": cfg_augmentation['augmentation_design']
                                }
            )
            config = wandb.config

        # run training
        scores, history = run_train(pref=pref, X_train_ir=X_train, y_train=y_train, X_test_ir=X_test, y_test=y_test, train_config=cfg_training,
                    model=model, use_wandb=wandb, augmentation=on_fly, class_weights=class_weights, fold_no=fold_no, training_mode='hyperparameter_tune')

        # store metrics for selecting the best values later
        val_loss_per_fold.append(scores[0])
        val_iou_per_fold.append(scores[1])

        if wandb:
            wandb.join()

        # increase fold number
        fold_no = fold_no + 1

    # determine best averaged run and store results in csv
    best = [a + b + c + d for a, b, c, d in zip(val_iou_per_fold[0], val_iou_per_fold[1], val_iou_per_fold[2], val_iou_per_fold[3])]
    best_epoch = max((v, i) for i, v in enumerate(best))[1]
    best_iou = (max((v, i) for i, v in enumerate(best))[0]) / 4

    print("Best epoch: ".format(best_epoch))
    print("Best IOU: ".format(best_iou))

    headers = ['best_avg_epoch_across_folds', 'best_avg_iou_across_folds']

    with open('metrics/hyperparameter_tune_results/{}.csv'.format(base_pref), 'a', newline='') as f:
        writer = csv.writer(f)
        # headers in the first row
        if f.tell() == 0:
            writer.writerow(headers)
        writer.writerow([best_epoch, best_iou])

    # provide average scores
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(val_iou_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {val_loss_per_fold[i]} - IoU: {val_iou_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> IoU: {np.mean(val_iou_per_fold)} (+- {np.std(val_iou_per_fold)})')
    print(f'> Loss: {np.mean(val_loss_per_fold)}')
    print('------------------------------------------------------------------------')
    print('Best run')
    print(f'Best averaged val_iou is {best_iou} in epoch {best_epoch}')


if __name__ == "__main__":
    main()

    



