import albumentations as A
import numpy as np
import random
import cv2

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


def get_training_augmentation(im_size, mode=0):
    """
    structure inspired by https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb
    
    Defines augmentation for training data. Each technique applied with a probability.
    
    Parameters:
    -----------
        mode : int
            defines methods used (for experimental reasons).
            one of 0: flip
                   1: rotate
                   2: crop
                   3: brightness, contrast
                   4: sharpen, blur
                   5: gaussian noise injection
    
    Return:
    -------
        train_transform : albumentations.compose
    """
    if mode == 0:
        train_transform = [
            A.HorizontalFlip(),
            A.VerticalFlip(),           
        ]
        return A.Compose(train_transform)
    
    elif mode == 1:
        train_transform = [
            # interpolation 0 means nearest interpolation such that mask labels are preserved
            A.Rotate(interpolation=0),
        ]
        return A.Compose(train_transform)

    elif mode == 2:
        train_transform = [
            A.RandomSizedCrop(min_max_height=[int(0.5*im_size), int(0.8*im_size)], height=im_size, width=im_size, interpolation=0, p=0.5),
        ]
        return A.Compose(train_transform)
    
    elif mode == 3:
        train_transform = [
            A.RandomBrightnessContrast(),
        ]
        return A.Compose(train_transform)
    
    elif mode == 4:
        train_transform = [
            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(p=1),
                    A.MotionBlur(p=1),
                ],
                p=0.5,
            ),
        ]
        return A.Compose(train_transform)

    elif mode == 5:
        train_transform = [
            A.GaussNoise(),
        ]
        return A.Compose(train_transform)
    
    elif mode == 6:
        train_transform = [
            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(p=1),
                    A.MotionBlur(p=1),
                ],
                p=1,
            ),
        ]
        return A.Compose(train_transform)


def get_preprocessing(preprocessing_fn):
    """
    Preprocessing function.
    
    Parameters:
    -----------
        preprocessing_fn : data normalization function 
            (can be specific for each pretrained neural network)
    
    Return:
    -------
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


def offline_augmentation(trainX, trainy, im_size, mode, factor=2):
  """
  Applies offline augmentation. Ds size will be increased by factor.

  Parameters:
  ----------
    trainX : np.ndarray
        original train images
    trainy : np.ndarray
        original train masks
    im_size : int
        image size
    mode : int
        experimental augmentation mode (see above)
    factor : int

  Return:
  -------
    trainX_new : np.ndarray
        new train images
    trainy_new : np.ndarray
        new train masks

  """

  im_aug_list = []
  ma_aug_list = []

  for i in range(0,factor-1):
    for idx in range(0,trainX.shape[0]):
        img = trainX[idx]
        msk = trainy[idx]
        aug = get_training_augmentation(im_size=im_size, mode=mode)
        sample = aug(image=img, mask=msk)
        im_aug, ma_aug = sample['image'], sample['mask']
        im_aug_list.append(im_aug)
        ma_aug_list.append(ma_aug)
  
  im_aug_np = np.array(im_aug_list)
  ma_aug_np = np.array(ma_aug_list)

  trainX_new = np.concatenate((trainX, im_aug_np), axis=0)
  trainy_new = np.concatenate((trainy, ma_aug_np), axis=0)

  return trainX_new, trainy_new

