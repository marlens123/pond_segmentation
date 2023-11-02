import sys
import os

# add parent directory to system path to be able to assess functions from root
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import pickle
import keras
from utils.augmentation import get_training_augmentation, get_preprocessing, offline_augmentation
from utils.train_helpers import patch_pipeline, patch_extraction, compute_class_weights
from utils.data import Dataloder, Dataset
from sklearn.model_selection import KFold
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
        WandbMetricsLogger()
    ]

    history = model.fit(train_dataloader,
                        verbose=1,
                        callbacks=callbacks,
                        steps_per_epoch=len(train_dataloader), 
                        epochs=num_epochs,  
                        validation_data=valid_dataloader, 
                        validation_steps=len(valid_dataloader),
                        shuffle=False)

    # save model scores
    with open('./scores/{}_trainHistoryDict'.format(pref), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # generalization metrics of trained model
    scores = model.evaluate(valid_dataloader, verbose=0)

    # history generalization metric
    hist_val_iou = history.history['val_mean_iou']
        
    return scores, hist_val_iou
    



def train_wrapper(X, y, im_size, pref, backbone='resnet34', loss='categoricalCE', freeze_tune=False,
              optimizer='Adam', train_transfer=None, encoder_freeze=False, input_normalize=False,
              batch_size=4, augmentation=None, mode=0, factor=2, epochs=100, patch_mode='slide_slide',
              weight_classes=False, use_dropout=False, use_batchnorm=True):
    """
    Function that starts the training pipeline for model selection (with 4-crossfold validation).

    Parameters:
    -----------
        X : numpy.ndarray
            images
        y : numpy.ndarray
            image labels
        im_size : int
            patch size
        base_pref : str
            identifier for training run
        backbone : str
        loss : str
        freeze_tune : Bool
            (doesn't work yet) if True, freezes encoder for half of epochs and sets to trainable for second half
        optimizer : str
        train_transfer : str or None
            'imagenet' or None (from scratch)
        encoder_freeze : Bool
            if True, uses fixed feature extractor when pre-training
        input_normalize : Bool
            (not used) whether to normalize input
        batch_size : int
        augmentation : str or None
            can be 'on_fly' or 'offline'
        mode : int
            augmentation mode - 0 = flipping, 1 = rotation, 2 = cropping, 3 = brightness contrast, 4 = sharpen blur, 5 = Gaussian noise
        factor : int
            used for offline augmentation: magnitude by which to increase dataset size
        epochs : int
        patch_mode : str
            (for this work, only 'slide_slide' is used) - whether to extract patches with sliding window ('slide_slide'), randomly ('random_random') 
            or training set in random and testing in slide mode ('random_slide')
        weight_classes : Bool
            whether to weight classes in loss function
        use_dropout : Bool
            whether to use dropout in decoder
        use_batchnorm : Bool
            (not used) whether to use batchnorm
    
    Return:
    ------
        time used (not used), fold statistics, and best average iou with corresponding epoch

    """

    ################################################################
    ##################### HYPERPARAMETER SETUP #####################
    ################################################################

    BACKBONE = backbone
    TRAIN_TRANSFER = train_transfer
    AUGMENTATION = augmentation
    BATCH_SIZE = batch_size

    if AUGMENTATION == 'on_fly':    
        on_fly = get_training_augmentation(im_size=im_size, mode=mode)
    else:
        on_fly = None

    if freeze_tune:
        encoder_freeze=True

    #################################################################
    ######################### MODEL SETUP ###########################
    #################################################################


    model = sm.Unet(BACKBONE, input_shape=(im_size, im_size, 3), classes=3, activation='softmax', encoder_weights=TRAIN_TRANSFER,
                    decoder_use_dropout=use_dropout, decoder_use_batchnorm=use_batchnorm, encoder_freeze=encoder_freeze)  

    print(type(model))
    print(model.summary())


    #################################################################
    ####################### CROSSFOLD SETUP #########################
    #################################################################

    num_folds = 4

    val_loss_per_fold = []
    val_iou_per_fold = []
    val_iou_weighted_per_fold = []
    val_f1_per_fold = []
    val_prec_per_fold = []
    val_rec_per_fold = []
    mp_per_class_per_fold = []
    si_per_class_per_fold = []
    oc_per_class_per_fold = []
    rounded_iou_per_fold = []

    time_per_fold = []             

    # define crossfold validator with random split
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=14)

    fold_no = 1
    fold_stats = []
    val_iou_all = []

    for train, test in kfold.split(X, y):

        # add fold number to prefix
        pref = pref + "_foldn{}".format(fold_no)

        ########################################## 
        ############## Class Weights #############
        ##########################################

        class_weights = compute_class_weights(y[train])
        
        ##########################################
        ############ Patch Extraction ############
        ##########################################

        if patch_mode=='random_random':
            X_train, y_train = patch_pipeline(X[train], y[train], patch_size=im_size)
            X_test, y_test = patch_pipeline(X[test], y[test], patch_size=im_size)

        elif patch_mode=='random_slide':
            X_train, y_train = patch_pipeline(X[train], y[train], patch_size=im_size)
            X_test, y_test = patch_extraction(X[test], y[test], size=im_size)

        elif patch_mode=='slide_slide':
            X_train, y_train = patch_extraction(X[train], y[train], size=im_size)
            X_test, y_test = patch_extraction(X[test], y[test], size=im_size)

        else:
            'Patch mode must be one of "random_random", "random_slide", "slide_slide"'


        fold_stats.append(y_test)

        print("Train size after patch extraction...", X_train.shape)
        print("Test size after patch extraction...", X_test.shape)

        ##########################################
        ######### Offline Augmentation ###########
        ##########################################

        if AUGMENTATION == 'offline':
            X_train, y_train = offline_augmentation(X_train, y_train, im_size=im_size, mode=mode, factor=factor)

        print("Train size imgs ...", X_train.shape)
        print("Train size masks ...", y_train.shape)
        print("Test size imgs ...", X_test.shape)
        print("Test size masks ...", y_test.shape)

        ##########################################
        ############# Tracking Config ############
        ##########################################

        run = wandb.init(project='pond_segmentation',
                            group=base_pref,
                            name='foldn_{}'.format(fold_no),
                            config={
                            "loss_function": loss,
                            "batch_size": batch_size,
                            "backbone": backbone,
                            "optimizer": optimizer,
                            "train_transfer": train_transfer,
                            "augmentation": AUGMENTATION
                            }
        )
        config = wandb.config

        print("Test set size...", X_test.shape)

        ##########################################
        ################ Training ################
        ##########################################

        scores, history = run_train(X_train, y_train, X_test, y_test, model=model, augmentation=on_fly, pref=pref, weight_classes=weight_classes, epochs=epochs,
                                    backbone=BACKBONE, batch_size=BATCH_SIZE, fold_no=fold_no, optimizer=optimizer, loss=loss, class_weights=class_weights,
                                    input_normalize=input_normalize, final_run=False, freeze_tune=freeze_tune)
        
        val_iou_all.append(history)

        val_loss_per_fold.append(scores[0])
        val_iou_per_fold.append(scores[1])
        val_iou_weighted_per_fold.append(scores[2])
        val_f1_per_fold.append(scores[3])
        val_prec_per_fold.append(scores[4])
        val_rec_per_fold.append(scores[5])
        mp_per_class_per_fold.append(scores[6])
        si_per_class_per_fold.append(scores[7])
        oc_per_class_per_fold.append(scores[8])
        rounded_iou_per_fold.append(scores[9])

        # close run for that fold
        wandb.join()

        # Increase fold number
        fold_no = fold_no + 1

    print(len(val_iou_all))
    # best averaged run
    best = [a + b + c + d for a, b, c, d in zip(val_iou_all[0], val_iou_all[1], val_iou_all[2], val_iou_all[3])]
    best_epoch = max((v, i) for i, v in enumerate(best))[1]
    best_iou = (max((v, i) for i, v in enumerate(best))[0]) / 4

    # == Provide average scores ==
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

    val_iou_per_fold = np.array(val_iou_per_fold)
    val_loss_per_fold = np.array(val_loss_per_fold)
    val_iou_weighted_per_fold = np.array(val_iou_weighted_per_fold)
    val_f1_per_fold = np.array(val_f1_per_fold)
    val_prec_per_fold = np.array(val_prec_per_fold)
    val_rec_per_fold = np.array(val_rec_per_fold)
    mp_per_class_per_fold = np.array(mp_per_class_per_fold)
    si_per_class_per_fold = np.array(si_per_class_per_fold)
    oc_per_class_per_fold = np.array(oc_per_class_per_fold)
    rounded_iou_per_fold = np.array(rounded_iou_per_fold)

    return fold_stats, (best_epoch, best_iou)
