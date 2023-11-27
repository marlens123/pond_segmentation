import sys
import os

# add parent directory to system path to be able to assess functions from root
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--date", default="220730_3", type=str, help="Date of the flight to be added.")
parser.add_argument("--flight_nr", default="flight_16", type=str, help="Flight number.")

parser.add_argument("--path_to_X_train", default="data/semi_super/train_images.npy", type=str, help="Path to current training images in .npy file format.")
parser.add_argument("--path_to_y_train", default="data/semi_super/train_masks.npy", type=str, help="Path to current training masks in .npy file format.")


def main():
    args = parser.parse_args()
    params = vars(args)

    indeces_flight7 = [103, 289, 600, 602, 208, 214, 230, 286, 298, 557, 593, 595, 598, 
               603, 617, 622, 621, 625, 650, 204, 206, 209, 236, 275, 345, 378, 
               535, 537, 567, 573, 605, 614, 616, 618, 619, 643, 205, 207, 211, 
               212, 213, 217, 215, 218, 228, 253, 276, 277, 629, 679, 680]
    
    indeces_flight9 = [618, 611, 610, 605, 602, 601, 692, 693, 700, 649, 652, 676, 678,
                679, 680, 689, 691, 694, 1005, 1006, 1009, 609, 594, 480, 479, 
                417, 305, 260, 208, 207, 199, 153, 152, 144, 129, 124, 121, 627, 
                640, 653, 656, 659, 661, 664, 667, 669, 668, 672, 682, 683, 684, 
                1012, 1013, 501, 614, 599, 598, 597, 596, 592, 488, 483, 481, 477, 
                473, 450, 431, 310, 308, 273, 250, 245, 231, 209, 206, 205, 158, 
                156, 155, 149, 128, 123, 120, 119, 117, 116, 114, 109, 633, 637, 
                665, 670, 671, 673, 674, 1008, 1010, 1011, 1014, 1015, 489, 472, 
                470, 461, 460, 457, 454, 449, 448, 442, 441, 429, 413, 412, 103, 96]
    
    indeces_flight10 = [325, 324, 323, 310, 309, 299, 298, 297, 289, 326, 319, 311, 308, 300, 
               292, 291, 288, 251, 268, 269, 277, 278, 285, 322, 320, 318, 317, 316, 
               306, 304, 303, 302, 301]
    
    indeces_flight11 = [218, 213, 110, 111]

    indeces = [312, 315, 531, 538, 1020, 1021, 1101, 1102, 309, 316, 527, 540, 552, 
               595, 606, 723, 1011, 1099, 1104, 1111]

    path = 'data/prediction/predicted/{}/raw/'.format(params['flight_nr'])

    masks_to_add = []
    imgs_to_add = []

    for idx in indeces:
        mask = cv2.imread(os.path.join(path, '{}.png'.format(idx)), 0)
        image = cv2.imread('data/prediction/preprocessed/{0}/{1}.png'.format(params['date'], idx*4), 0)

        imgs_to_add.append(image)
        masks_to_add.append(mask)

    masks_to_add = np.array(masks_to_add)
    imgs_to_add = np.array(imgs_to_add)

    images = np.load(params['path_to_X_train'])
    masks = np.load(params['path_to_y_train'])

    new_images = np.concatenate((images, imgs_to_add), axis=0)
    new_masks = np.concatenate((masks, masks_to_add), axis=0)

    print("New shape images: {}".format(new_images.shape))
    print("New shape masks: {}".format(new_masks.shape))

    np.save('data/semi_super/train_images.npy', new_images)
    np.save('data/semi_super/train_masks.npy', new_masks)

if __name__ == "__main__":
    main()