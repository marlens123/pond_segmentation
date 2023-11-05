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

#import wandb
#wandb.login()

parser = argparse.ArgumentParser(description="Model fine-tuning. Default hyperparameter values were optimized during previous experiments.")

# prefix
parser.add_argument("--pref", default="ho_001", type=str, help="Identifier for the run. Model scores will be stored with this prefix.")

# data
parser.add_argument("--X", default="data/training/train_images.npy", type=str, help="Path to training images in .npy file format.")
parser.add_argument("--y", default="data/training/train_masks.npy", type=str, help="Path to training masks in .npy file format.")

# hyperparameters
parser.add_argument("--im_size", default=480, type=int, choices=[32, 64, 128, 256, 480], help="Patch size to train on. Choices are constrained because of patch extraction setup.")
parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs. The weights of the best performing training epoch will be stored.")
parser.add_argument("--loss", default="focal_dice", type=str, choices=["categoricalCE", "focal_dice", "focal"], help="Loss function. E.g. 'categorical_CE' or 'focal_dice'. For more options see sm.")
parser.add_argument("--backbone", default="resnet34", type=str, help="U-net backbone to use. For options see sm.")
parser.add_argument("--optimizer", default="Adam", type=str, choices=["Adam", "SGD", "Adamax"], help="Optimizer to use. For options see sm.")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size. Adjust with respect to training set size and patch size.")
parser.add_argument("--augmentation_design", default="on_fly", type=str, choices=["none", "offline", "on_fly"], help="Either None, 'offline' (fixed augmentation before training), or 'on_fly' (while feeding data into the model).")
parser.add_argument("--augmentation_technique", default=4, type=int, choices=[0, 1, 2, 3, 4, 5], help="0 : flip, 1 : rotate, 2 : crop, 3 : brightness contrast, 4 : sharpen blur, 5 : Gaussian noise.")
parser.add_argument("--augmentation_factor", default=2, type=int, help="Magnitude by which the dataset will be increased through augmentation. Only takes effect when augmentation_design is set 'offline'.")
parser.add_argument("--use_class_weights", action='store_true', help="If the loss function should account for class imbalance.")
parser.add_argument("--use_dropout", action='store_true', help="If to use dropout layers after upsampling operations in the decoder.")
parser.add_argument("--pretrain", default="imagenet", type=str, choices=["imagenet", "none"], help="Either 'imagenet' to use encoder weights pretrained on ImageNet or None to train from scratch.")
parser.add_argument("--freeze", action='store_true', help="Only takes effect when pretrain is not None. Whether to freeze encoder during training or allow fine-tuning of encoder weights.")


def main():
    args = parser.parse_args()
    params = vars(args)

    # load data
    X = np.load(params['X'])
    y = np.load(params['y'])

    # set augmentation
    on_fly = None
    if params['augmentation_design'] == 'on_fly':
        on_fly = get_training_augmentation(im_size=params['im_size'], mode=params['augmentation_technique'])

    # set pretraining
    if params['pretrain'] == "none":
        params['pretrain'] = None

    # construct model
    model = sm.Unet(params['backbone'], input_shape=(params['im_size'], params['im_size'], 3), classes=3, activation='softmax', encoder_weights=params['pretrain'],
                    decoder_use_dropout=params['use_dropout'], encoder_freeze=params['freeze'])  

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
        if params['use_class_weights']:
            class_weights = compute_class_weights(y[train])
            print("Class weights are...:", class_weights)
        else:
            class_weights = None

        # patch extraction
        X_train, y_train = patch_extraction(X[train], y[train], size=params['im_size'])
        X_test, y_test = patch_extraction(X[test], y[test], size=params['im_size'])

        # offline augmentation if selected
        if params['augmentation_design'] == 'offline':
            X_train, y_train = offline_augmentation(X_train, y_train, im_size=params['im_size'], mode=params['augmentation_technique'], factor=params['augmentation_factor'])

        """
        # tracking configuration
        run = wandb.init(project='pond_segmentation',
                            group=params['pref'],
                            name='foldn_{}'.format(fold_no),
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
        """

        # run training
        scores, history = run_train(pref=pref, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_epochs=params['num_epochs'],
                    loss=params['loss'], backbone=params['backbone'], optimizer=params['optimizer'], batch_size=params['batch_size'], 
                    model=model, augmentation=on_fly, class_weights=class_weights, fold_no=fold_no, training_mode='hyperparameter_tune')

        # store metrics for selecting the best values later
        val_loss_per_fold.append(scores[0])
        val_iou_per_fold.append(scores[1])

        # close tracking for that fold
        #wandb.join()

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

    



