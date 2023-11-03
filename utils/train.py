import sys
import os

# add parent directory to system path to be able to assess functions from root
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import pickle
import keras
import tensorflow as tf
import pandas as pd
from utils.augmentation import get_preprocessing
from utils.data import Dataloder, Dataset

import models.segmentation_models_qubvel as sm

import wandb
wandb.login()
from wandb.keras import WandbMetricsLogger


def run_train(pref, X_train, y_train, X_test, y_test, num_epochs, loss, backbone, optimizer, batch_size,
                model, augmentation=None, class_weights=None, fold_no=None):
    """
    Training function.

    Parameters:
    -----------
        pref : str
            identifier for training run
        X_train : numpy.ndarray
            train images
        y_train : numpy.ndarray
            image labels
        X_test : numpy.ndarray
            test images
        y_test : numpy.ndarray
            test labels
        num_epochs : int
            number of epochs
        loss : str
        backbone : str
        optimizer : str
        batch_size : int
        model : keras.engine.functional.Functional
            model defined before call
        augmentation : albumentations.core.composition.Compose
            specifies on-fly augmentation methods (if to be appplied; else None)
        class_weights : list
            the class weights to use (None when no weights should be used)
        fold_no : int
            if in hyperparameter optimization, number of the crossfold run
        final_run : Bool
            whether this is the final run (without crossfold validation)

    Return:
    ------
        scores, hist_val_iou   
            generalization metrics and history of generalization metrics
    """
    
    CLASSES=['melt_pond', 'sea_ice']
    weights = class_weights
    
    # training dataset
    train_dataset = Dataset(
        X_train, 
        y_train, 
        classes=CLASSES, 
        augmentation=augmentation,
        preprocessing=get_preprocessing(sm.get_preprocessing(backbone)),
    )

    # validation dataset
    valid_dataset = Dataset(
        X_test, 
        y_test, 
        classes=CLASSES,
        preprocessing=get_preprocessing(sm.get_preprocessing(backbone)),
    )

    train_dataloader = Dataloder(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    # define loss
    if loss == 'jaccard':
        LOSS = sm.losses.JaccardLoss(class_weights=weights)
    elif loss == 'focal_dice':
        dice_loss = sm.losses.DiceLoss(class_weights=weights) 
        focal_loss = sm.losses.CategoricalFocalLoss()
        LOSS = dice_loss + (1 * focal_loss)
    elif loss == 'categoricalCE':
        LOSS = sm.losses.CategoricalCELoss(class_weights=weights)
    elif loss== 'focal':
        LOSS = sm.losses.CategoricalFocalLoss()
    else:
        print('No loss function specified')

    # define optimizer
    if optimizer == 'Adam':
        OPTIMIZER = keras.optimizers.Adam()
    elif optimizer == 'SGD':
        OPTIMIZER = keras.optimizer.SGD()
    elif optimizer == 'Adamax':
        OPTIMIZER = keras.optimizer.Adamax()
    else:
        print('No optimizer specified')

    # define evaluation metrics
    mean_iou = sm.metrics.IOUScore(name='mean_iou')
    weighted_iou = sm.metrics.IOUScore(class_weights=class_weights, name='weighted_iou')
    f1 = sm.metrics.FScore(beta=1, name='f1')
    precision = sm.metrics.Precision(name='precision')
    recall = sm.metrics.Recall(name='recall')
    melt_pond_iou = sm.metrics.IOUScore(class_indexes=0, name='melt_pond_iou')
    sea_ice_iou = sm.metrics.IOUScore(class_indexes=1, name='sea_ice_iou')
    ocean_iou = sm.metrics.IOUScore(class_indexes=2, name='ocean_iou')
    rounded_iou = sm.metrics.IOUScore(threshold=0.5, name='mean_iou_rounded')

    # compile model
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[mean_iou, weighted_iou, f1, precision, recall, melt_pond_iou,
                                                           sea_ice_iou, ocean_iou, rounded_iou])

    # save weights of best performing model in terms of minimal val_loss
    callbacks = [
        keras.callbacks.ModelCheckpoint('./weights/best_model{}.h5'.format(pref), save_weights_only=True, save_best_only=True, mode='min'),
        WandbMetricsLogger(),
        tf.keras.callbacks.CSVLogger('./metrics/{}.csv'.format(pref))
    ]

    history = model.fit(train_dataloader,
                        verbose=1,
                        callbacks=callbacks,
                        steps_per_epoch=len(train_dataloader), 
                        epochs=num_epochs,  
                        validation_data=valid_dataloader, 
                        validation_steps=len(valid_dataloader),
                        shuffle=False)