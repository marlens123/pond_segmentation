import sys
import os

# add parent directory to system path to be able to assess functions from root
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import cv2

# code to add good performing predictions to training set
# identify indeces of good performing images

flight_nr = None

pref1 = None
pref2 = None

idx = None

ma = cv2.imread('data/prediction/predicted/{0}/raw/{1}'.format(pref1, idx), 0)
mask = ma.reshape(1, ma.shape)

# manually inspect which prediction is better: pref1 or pref2

# concatenate mask to existing mask file
# concatenate corresponding image to existing image file

im = cv2.imread('data/prediction/preprocessed/{0}/{1}'.format(flight_nr, idx), 0)
image = im.reshape(1, im.shape)

images = np.load('data/training/train_images.npy')
masks = np.load('data//training/train_masks.npy')

np.concatenate((images, image), axis=0)
np.concatenate((masks, mask), axis=0)