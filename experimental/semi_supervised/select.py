"""
MISSING:

- pass flight number
- goodness measure that determines if a given pseudo-label will be added to training data

"""



import sys
import os

# add parent directory to system path to be able to assess functions from root
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import cv2
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--iter_nr", default="1", type=str, help="Iteration number.")

parser.add_argument("--path_to_X_train", default="data/training/train_images.npy", type=str, help="Path to current training images in .npy file format.")
parser.add_argument("--path_to_y_train", default="data/training/train_masks.npy", type=str, help="Path to current training masks in .npy file format.")
parser.add_argument("--pref1", default="unet_flight9_flight16", type=str, help="Prefix of first model.")
parser.add_argument("--pref2", default="att_unet_flight9_flight16", type=str, help="Prefix of second model.")


def main():
    args = parser.parse_args()
    params = vars(args)

    pred1_path = 'data/prediction/predicted/{}/raw/'.format(params['pref1'])
    pred2_path = 'data/prediction/predicted/{}/raw/'.format(params['pref2'])

    masks_to_add = []
    masks_indeces = []

    for idx, file in enumerate(os.listdir(pred1_path)):
        pred1 = cv2.imread((os.path.join(pred1_path), file), 0)
        pred2 = cv2.imread((os.path.join(pred2_path), file), 0)

        goodness_value = None
        thrs = None

        # compare
        if goodness_value > thrs:
            # manually compare pred1 pred2 and replace best
            best = None

            masks_to_add.append(best)
            masks_indeces.append(file)

    if len(masks_to_add) == 0:
        print("No images to add to the training set")

    else:
        print("Training set size will be increased by {} images".format(len(masks_to_add)))

        images = np.load(params['path_to_X_train'])
        masks = np.load(params['path_to_y_train'])

        for idx, elem in enumerate(masks_to_add):
            mask = elem.reshape(1, elem.shape)

            image = cv2.imread('data/prediction/preprocessed/{0}/{1}'.format(flight_nr, masks_indeces[idx]), 0)
            image = image.reshape(1, image.shape)

            new_images = np.concatenate((images, image), axis=0)
            new_masks = np.concatenate((masks, mask), axis=0)

            print("New shape images: {}".format(images))
            print("New shape masks: {}".format(masks))

            np.save('data/semi_super/{}/train_images.npy'.format(params['iter_nr']), new_images)
            np.save('data/semi_super/{}/train_masks.npy'.format(params['iter_nr']), new_masks)
    

if __name__ == "__main__":
    main()
