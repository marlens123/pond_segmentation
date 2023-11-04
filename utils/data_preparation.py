import pandas as pd
from preprocess_helpers import crop_center_square
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from datetime import timedelta



def extract_time(img_idx, timestamps, flight_nr):
    """
    Convert the timestamp for an image.

    Parameters:
    ----------
        img_idx : int
        timestamps : numpy.ma.core.MaskedArray
    """
    
    if flight_nr == 9:
        reference = '2022-07-18 00:00:00'
    elif flight_nr == 16:
        reference = '2022-07-30 00:00:00'

    times = pd.Series(timestamps)
    date = pd.Timestamp(reference)
    time = str(date + timedelta(seconds=int(times[img_idx])))

    return time



def extract_single(dataset, idx, time, flight_nr, save_path):
    # extracts image in squared shape

    timestamp = extract_time(idx, time, flight_nr).replace(' ','_').replace(':','').replace('-','')
    img = dataset[idx]
    img = crop_center_square(img)

    # clip for better visibility, use 272,274 for flight 16
    img = np.clip(img, 273, 276)

    plt.imsave(os.path.join(save_path, '{}_{}.png'.format(timestamp,idx)), img, cmap='cividis')


def extract_without_overlap(dataset, dataset_size, time, flight_nr, save_path, clip=False):
    """
    Extracts only every fourth image - extracted images will be non-overlapping, saves memory.

    Parameters:
    -----------
        dataset : numpy.ma.core.MaskedArray
        dataset_size : int
        time : numpy.ma.core.MaskedArray
    """
    idx = 0
    
    for i in range(dataset_size):
        if(i % 4 == 0):
            timestamp = extract_time(i, time, flight_nr).replace(' ','_').replace(':','').replace('-','')
            img = dataset[i]

            # optionally clip for better visibility
            if clip:
                img = np.clip(img, 273, 276)

            plt.imsave(os.path.join(save_path, '{}_{}.png'.format(timestamp,idx)), img, cmap='cividis')

            idx = idx + 1


def extract_all(dataset, dataset_size, time, flight_nr, save_path):
    """
    Extracts all images for flight specified.

    Parameters:
    -----------
        dataset : numpy.ma.core.MaskedArray
        dataset_size : int
        time : numpy.ma.core.MaskedArray
    """
    idx = 0
    
    for i in range(dataset_size):
        timestamp = extract_time(i, time, flight_nr).replace(' ','_').replace(':','').replace('-','')
        img = dataset[i]

        # clip for better visibility
        img = np.clip(img, 273, 276)

        plt.imsave(os.path.join(save_path, '{}_{}.png'.format(timestamp,idx)), img, cmap='cividis')

        idx = idx + 1



def create_mask(image, threshold='li'):
    """
    Thresholds image and denoises with Scharr filter

    Parameters:
    -----------
        image : np.array
            source image for edge map
        threshold : str
            threshold to apply

    """
    if threshold == 'otsu':
        thrs = filters.threshold_otsu(image)
    elif threshold == 'mean':
        thrs = filters.threshold_mean(image)
    elif threshold == 'li':
        thrs = filters.threshold_li(image)
    
    binary = image > thrs

    img_blurred = filters.scharr(binary)

    plt.imshow(img_blurred, cmap='Greys')
    plt.imsave('E:/polar/code/ponds_extended_data/candidates/100.png', img_blurred, cmap='Greys')