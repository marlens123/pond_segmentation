import sys
import os

# add parent directory to system path to be able to assess functions from root
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import cv2
from sklearn.model_selection import train_test_split


# prepend test images
train_X = np.load('./data/semi_super/train_images.npy')
test_X = np.load('./data/training/flight9_flight16/test_images.npy')

X = np.concatenate((train_X, test_X), axis=0)

train_y = np.load('./data/semi_super/train_masks.npy')
test_y =np.load('./data/training/flight9_flight16/test_masks.npy')

y = np.concatenate((train_y, test_y), axis=0)

# array or list with sample weights

# manually chosen
sample_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.9, 0.9, 0.9, 0.85, 0.85, 0.85, 
                  0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.7, 0.7, 0.7, 
                  0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 
                  0.9, 0.9, 0.9, 0.9, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 
                  0.85, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 
                  0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 
                  0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 
                  0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 
                  0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.85, 0.85, 0.85, 
                  0.85, 0.85, 0.85, 0.85, 0.85, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 1, 1]

print(len(sample_weights))
print(X.shape)
print(y.shape)

# train test split both
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=4)

weights_train, weights_test = train_test_split(sample_weights, test_size=0.15, random_state=4)

np.save('data/semi_super/X_train.npy', X_train)
np.save('data/semi_super/y_train.npy', y_train)
np.save('data/semi_super/X_test.npy', X_test)
np.save('data/semi_super/y_test.npy', y_test)

np.save('data/semi_super/weights_train.npy', np.array(weights_train))
np.save('data/semi_super/weights_test.npy', np.array(weights_test))