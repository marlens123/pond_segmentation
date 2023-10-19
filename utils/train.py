import sys
import os

# add parent directory to system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import tensorflow as tf
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import keras
from utils.augmentation import get_training_augmentation, get_preprocessing, offline_augmentation
from utils.patch_extraction import patch_pipeline, patch_extraction
from utils.data import Dataloder, Dataset
from sklearn.model_selection import KFold
import models.segmentation_models_qubvel as sm
from models.segmentation_models_qubvel.segmentation_models.utils import set_trainable

#import wandb
#wandb.init(project="extended")
#from wandb.keras import WandbMetricsLogger
#wandb.login()

from timeit import default_timer as timer

# inspiration: https://stackoverflow.com/questions/57181551/can-i-write-a-keras-callback-that-records-and-returns-the-total-training-time
class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)


def run_train(X_train, y_train, X_test, y_test, model, pref, backbone='resnet34', batch_size=4, weight_classes=False, epochs=100,
              class_weights=None, loss='categoricalCE', optimizer='Adam', augmentation=None, input_normalize=False, fold_no=None, final_run=False, freeze_tune=False,
              early_stop=False):
    """
    Training function.

    Parameters:
    -----------
        X_train : numpy.ndarray
            train images
        y_train : numpy.ndarray
            image labels
        X_test : numpy.ndarray
            test images
        y_test : numpy.ndarray
            test labels
        model : 
        pref : str
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
        fold_no : int
            number of the crossfold run
        final_run : Bool
            whether this is the final run (without crossfold validation)
        batch_size : int
        augmentation : 
            on-fly augmentation methods (if to be appplied; else None)
        final_run : Bool
            (not used) whether this is the final run
        epochs : int
        weight_classes : Bool
            whether to weight classes in loss function
        class_weights :
            the class weights to use
    
    Return:
    ------
        model, scores, hist_val_iou, time   
    """
    
    CLASSES=['melt_pond', 'sea_ice']
    BACKBONE = backbone
    BATCH_SIZE = batch_size

    if weight_classes:
        weights = class_weights
    else:
        weights = None
    
    # Dataset for train images
    train_dataset = Dataset(
        X_train, 
        y_train, 
        classes=CLASSES, 
        normalize=input_normalize,
        augmentation=augmentation,
        preprocessing=get_preprocessing(sm.get_preprocessing(BACKBONE)),
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        X_test, 
        y_test, 
        classes=CLASSES,
        normalize=input_normalize, 
        preprocessing=get_preprocessing(sm.get_preprocessing(BACKBONE)),
    )

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

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

    if optimizer == 'Adam':
        OPTIMIZER = keras.optimizers.Adam()
    elif optimizer == 'SGD':
        OPTIMIZER = keras.optimizer.SGD()
    elif optimizer == 'Adamax':
        OPTIMIZER = keras.optimizer.Adamax()
    else:
        print('No optimizer specified')

    mean_iou = sm.metrics.IOUScore(name='mean_iou')
    weighted_iou = sm.metrics.IOUScore(class_weights=class_weights, name='weighted_iou')
    f1 = sm.metrics.FScore(beta=1, name='f1')
    precision = sm.metrics.Precision(name='precision')
    recall = sm.metrics.Recall(name='recall')
    melt_pond_iou = sm.metrics.IOUScore(class_indexes=0, name='melt_pond_iou')
    sea_ice_iou = sm.metrics.IOUScore(class_indexes=1, name='sea_ice_iou')
    ocean_iou = sm.metrics.IOUScore(class_indexes=2, name='ocean_iou')
    rounded_iou = sm.metrics.IOUScore(threshold=0.5, name='mean_iou_rounded')


    # threshold value in iou metric will round predictions
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[mean_iou, weighted_iou, f1, precision, recall, melt_pond_iou,
                                                           sea_ice_iou, ocean_iou, rounded_iou])


    if early_stop:
        # save weights of best performing model in terms of minimal val_loss
        callbacks = [
            keras.callbacks.ModelCheckpoint('./weights/best_model{}.h5'.format(pref), save_weights_only=True, save_best_only=True, mode='min'),
            keras.callbacks.EarlyStopping(patience=10),
            TimingCallback(),
            #WandbMetricsLogger()
        ]

    else:
        # save weights of best performing model in terms of minimal val_loss
        callbacks = [
            keras.callbacks.ModelCheckpoint('./weights/best_model{}.h5'.format(pref), save_weights_only=True, save_best_only=True, mode='min'),
            TimingCallback(),
            #WandbMetricsLogger()
        ]


    if freeze_tune:
        history = model.fit(train_dataloader,
                            verbose=1,
                            callbacks=callbacks,
                            steps_per_epoch=len(train_dataloader), 
                            epochs=50,  
                            validation_data=valid_dataloader, 
                            validation_steps=len(valid_dataloader),
                            shuffle=False)        

        # unfreeze encoder
        set_trainable(model)

        history = model.fit(train_dataloader,
                            verbose=1,
                            callbacks=callbacks,
                            steps_per_epoch=len(train_dataloader), 
                            epochs=50,  
                            validation_data=valid_dataloader, 
                            validation_steps=len(valid_dataloader),
                            shuffle=False) 

    else:
        history = model.fit(train_dataloader,
                            verbose=1,
                            callbacks=callbacks,
                            steps_per_epoch=len(train_dataloader), 
                            epochs=epochs,  
                            validation_data=valid_dataloader, 
                            validation_steps=len(valid_dataloader),
                            shuffle=False)

    # save model scores
    with open('./scores/{}_trainHistoryDict'.format(pref), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    time = callbacks[1].logs

    # generalization metrics of trained model
    scores = model.evaluate(valid_dataloader, verbose=0)

    # history generalization metric
    hist_val_iou = history.history['val_mean_iou']
        
    return model, scores, hist_val_iou, time
    



def train_wrapper(X, y, im_size, base_pref, backbone='resnet34', loss='categoricalCE', freeze_tune=False,
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

    #dot_img_file = './summary/model_dropout.png'
    #tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, dpi=300)


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

        ##########################################
        ################# Prefix #################
        ##########################################

        pref = base_pref + "_foldn{}".format(fold_no)

        ########################################## 
        ############## Class Weights #############
        ##########################################

        masks_resh = y[train].reshape(-1,1)
        masks_resh_list = masks_resh.flatten().tolist()
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(masks_resh), y=masks_resh_list)
        print("Class weights are...:", class_weights)
        
        ##########################################
        ############ Patch Extraction ############
        ##########################################

        # 320 random patches per image
        if im_size==32:
            if patch_mode=='random_random':
                X_train, y_train = patch_pipeline(X[train], y[train], nr_patches=320, patch_size=32)
                X_test, y_test = patch_pipeline(X[test], y[test], nr_patches=320, patch_size=32)

            elif patch_mode=='random_slide':
                X_train, y_train = patch_pipeline(X[train], y[train], nr_patches=320, patch_size=32)
                X_test, y_test = patch_extraction(X[test], y[test], size=32, step=32)

            elif patch_mode=='slide_slide':
                X_train, y_train = patch_extraction(X[train], y[train], size=32, step=32)
                X_test, y_test = patch_extraction(X[test], y[test], size=32, step=32)

            else:
                'Patch mode must be one of "random_random", "random_slide", "slide_slide"'
        
        # 80 random patches per image
        elif im_size==64:
            if patch_mode=='random_random':
                X_train, y_train = patch_pipeline(X[train], y[train], nr_patches=80, patch_size=64)
                X_test, y_test = patch_pipeline(X[test], y[test], nr_patches=80, patch_size=64)

            elif patch_mode=='random_slide':
                X_train, y_train = patch_pipeline(X[train], y[train], nr_patches=80, patch_size=64)
                X_test, y_test = patch_extraction(X[test], y[test], size=64, step=68)

            elif patch_mode=='slide_slide':
                X_train, y_train = patch_extraction(X[train], y[train], size=64, step=68)
                X_test, y_test = patch_extraction(X[test], y[test], size=64, step=68)

            else:
                'Patch mode must be one of "random_random", "random_slide", "slide_slide"'
        
        # 20 random patches per image
        elif im_size==128:
            if patch_mode=='random_random': 
                X_train, y_train = patch_pipeline(X[train], y[train], nr_patches=20, patch_size=128)
                X_test, y_test = patch_pipeline(X[test], y[test], nr_patches=20, patch_size=128)

            elif patch_mode=='random_slide':
                X_train, y_train = patch_pipeline(X[train], y[train], nr_patches=20, patch_size=128)
                X_test, y_test = patch_extraction(X[test], y[test], size=128, step=160)

            elif patch_mode=='slide_slide':
                X_train, y_train = patch_extraction(X[train], y[train], size=128, step=160)
                X_test, y_test = patch_extraction(X[test], y[test], size=128, step=160)

            else:
                'Patch mode must be one of "random_random", "random_slide", "slide_slide"'
        
        # 5 random patches per image
        elif im_size==256:
            if patch_mode=='random_random':
                X_train, y_train = patch_pipeline(X[train], y[train], nr_patches=5, patch_size=256)
                X_test, y_test = patch_pipeline(X[test], y[test], nr_patches=5, patch_size=256)

            elif patch_mode=='random_slide':
                X_train, y_train = patch_pipeline(X[train], y[train], nr_patches=5, patch_size=256)
                X_test, y_test = patch_extraction(X[test], y[test], size=256, step=224)

            elif patch_mode=='slide_slide':
                X_train, y_train = patch_extraction(X[train], y[train], size=256, step=224)
                X_test, y_test = patch_extraction(X[test], y[test], size=256, step=224)

            else:
                'Patch mode must be one of "random_random", "random_slide", "slide_slide"'

        # no patch extraction
        elif im_size==480:
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]

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
        """
        run = wandb.init(project='extended',
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
        """

        ##########################################
        ################ Training ################
        ##########################################

        model, scores, history, time = run_train(X_train, y_train, X_test, y_test, model=model, augmentation=on_fly, pref=pref, weight_classes=weight_classes, epochs=epochs,
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

        # sum up training time for individual epochs
        time_per_fold.append(sum(time))

        # close run for that fold
        #wandb.join()

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

    time_per_fold = np.array(time_per_fold)

    return time_per_fold, fold_stats, (best_epoch, best_iou)


def final_train(X_train, y_train, X_test, y_test, im_size, base_pref, backbone='resnet34', loss='categoricalCE', freeze_tune=False,
                optimizer='Adam', train_transfer=None, encoder_freeze=False, input_normalize=False,
                batch_size=4, augmentation=None, mode=0, factor=2, epochs=100, patch_mode='slide_slide',
                weight_classes=False, use_dropout=False, use_batchnorm=True, early_stop=False):
    """
    Function that starts the training pipeline for final model construction (with simple train-test split).

    Parameters:
    -----------
        X_train : numpy.ndarray
            train images
        y_train : numpy.ndarray
            train image labels
        X_test : numpy.ndarray
            test images
        y_test : numpy.ndarray
            test labels
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
        early_stop : Bool
            (not used) whether to use early stopping
    
    Return:
    ------
        time used (not used)

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

    print(model.summary())           

    ##########################################
    ################# Prefix #################
    ##########################################

    # prefix should contain the fold number
    pref = base_pref

    ########################################## 
    ############## Class Weights #############
    ##########################################

    masks_resh = y_train.reshape(-1,1)
    masks_resh_list = masks_resh.flatten().tolist()
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(masks_resh), y=masks_resh_list)
    print("Class weights are...:", class_weights)
    
    ##########################################
    ############ Patch Extraction ############
    ##########################################

    # 320 random patches per image
    if im_size==32:
        if patch_mode=='random_random':
            X_train, y_train = patch_pipeline(X_train, y_train, nr_patches=320, patch_size=32)
            X_test, y_test = patch_pipeline(X_test, y_test, nr_patches=320, patch_size=32)

        elif patch_mode=='random_slide':
            X_train, y_train = patch_pipeline(X_train, y_train, nr_patches=320, patch_size=32)
            X_test, y_test = patch_extraction(X_test, y_test, size=32, step=32)

        elif patch_mode=='slide_slide':
            X_train, y_train = patch_extraction(X_train, y_train, size=32, step=32)
            X_test, y_test = patch_extraction(X_test, y_test, size=32, step=32)

        else:
            'Patch mode must be one of "random_random", "random_slide", "slide_slide"'
    
    # 80 random patches per image
    elif im_size==64:
        if patch_mode=='random_random':
            X_train, y_train = patch_pipeline(X_train, y_train, nr_patches=80, patch_size=64)
            X_test, y_test = patch_pipeline(X_test, y_test, nr_patches=80, patch_size=64)

        elif patch_mode=='random_slide':
            X_train, y_train = patch_pipeline(X_train, y_train, nr_patches=80, patch_size=64)
            X_test, y_test = patch_extraction(X_test, y_test, size=64, step=68)

        elif patch_mode=='slide_slide':
            X_train, y_train = patch_extraction(X_train, y_train, size=64, step=68)
            X_test, y_test = patch_extraction(X_test, y_test, size=64, step=68)

        else:
            'Patch mode must be one of "random_random", "random_slide", "slide_slide"'
        
    # 20 random patches per image
    elif im_size==128:
        if patch_mode=='random_random': 
            X_train, y_train = patch_pipeline(X_train, y_train, nr_patches=20, patch_size=128)
            X_test, y_test = patch_pipeline(X_test, y_test, nr_patches=20, patch_size=128)

        elif patch_mode=='random_slide':
            X_train, y_train = patch_pipeline(X_train, y_train, nr_patches=20, patch_size=128)
            X_test, y_test = patch_extraction(X_test, y_test, size=128, step=160)

        elif patch_mode=='slide_slide':
            X_train, y_train = patch_extraction(X_train, y_train, size=128, step=160)
            X_test, y_test = patch_extraction(X_test, y_test, size=128, step=160)

        else:
            'Patch mode must be one of "random_random", "random_slide", "slide_slide"'
        
    # 5 random patches per image
    elif im_size==256:
        if patch_mode=='random_random':
            X_train, y_train = patch_pipeline(X_train, y_train, nr_patches=5, patch_size=256)
            X_test, y_test = patch_pipeline(X_test, y_test, nr_patches=5, patch_size=256)

        elif patch_mode=='random_slide':
            X_train, y_train = patch_pipeline(X_train, y_train, nr_patches=5, patch_size=256)
            X_test, y_test = patch_extraction(X_test, y_test, size=256, step=224)

        elif patch_mode=='slide_slide':
            X_train, y_train = patch_extraction(X_train, y_train, size=256, step=224)
            X_test, y_test = patch_extraction(X_test, y_test, size=256, step=224)

        else:
            'Patch mode must be one of "random_random", "random_slide", "slide_slide"'

    # no patch extraction
    elif im_size==480:
        X_train = X_train
        y_train = y_train
        X_test = X_test
        y_test = y_test


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
    """
    run = wandb.init(project='extended',
                        group=base_pref,
                        name=base_pref,
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
    """

    print("Test set size...", X_test.shape)

    ##########################################
    ################ Training ################
    ##########################################

    model, scores, history, time = run_train(X_train, y_train, X_test, y_test, model=model, augmentation=on_fly, pref=pref, weight_classes=weight_classes, epochs=epochs,
                                backbone=BACKBONE, batch_size=BATCH_SIZE, fold_no=0, optimizer=optimizer, loss=loss, class_weights=class_weights,
                                input_normalize=input_normalize, final_run=True, freeze_tune=freeze_tune, early_stop=early_stop)
    
    #wandb.join()

    val_iou_all = history
    time_list = []

    val_loss = scores[0]
    val_iou = scores[1]
    val_iou_weighted = scores[2]
    val_f1 = scores[3]
    val_prec = scores[4]
    val_rec = scores[5]
    mp_per_class = scores[6]
    si_per_class = scores[7]
    oc_per_class = scores[8]
    rounded_iou = scores[9]

    # sum up training time for individual epochs
    time_list.append(sum(time))

    print(len(val_iou_all))

    val_iou = np.array(val_iou)
    val_loss = np.array(val_loss)
    val_iou_weighted = np.array(val_iou_weighted)
    val_f1 = np.array(val_f1)
    val_prec = np.array(val_prec)
    val_rec = np.array(val_rec)
    mp_per_class = np.array(mp_per_class)
    si_per_class = np.array(si_per_class)
    oc_per_class = np.array(oc_per_class)
    rounded_iou = np.array(rounded_iou)

    time_array = np.array(time_list)

    return time_array
