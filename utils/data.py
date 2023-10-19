# classes for data loading and preprocessing
# Inspired by https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb 

import os
import cv2
import numpy as np
import keras

def expand_greyscale_channels(image):
    # add channel dimension
    image = np.expand_dims(image, -1)
    # copy last dimension to reach shape of RGB
    image = image.repeat(3, axis=-1)
    return image

class Dataset:
    """Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['melt_pond', 'sea_ice', 'ocean']
    
    def __init__(
            self, 
            images, 
            masks, 
            classes=None,
            augmentation=None, 
            preprocessing=None,
            normalize=False,
    ):
        self.images_fps = images.tolist()
        self.masks_fps = masks.tolist()
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.normalize = normalize

        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data as grayscale
        image = self.images_fps[i]
        image = np.array(image)
        # reshape to 3 dims in last channel
        image = expand_greyscale_channels(image)

        mask = self.masks_fps[i]
        mask = np.array(mask)
        #mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        
        # one-hot encoding of masks
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        background = 1 - mask.sum(axis=-1, keepdims=True)
        mask = np.concatenate((mask, background), axis=-1)

        print(mask.shape)
        print(np.unique(mask))

        print(image.dtype)

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        # optional normalization (not used in thesis)
        print("Image shape before normalization...", image.shape)
        if self.normalize:
            image = image / 255
        print("Image shape after normalization...", image.shape)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        print("Shape image: {}".format(image.shape))
        print("Shape mask: {}".format(mask.shape))

        return image, mask
        
    def __len__(self):
        return len(self.images_fps)
    
    
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
            
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)