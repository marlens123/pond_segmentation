import numpy as np
import cv2
import os
from skimage.transform import resize
import matplotlib.pyplot as plt

def transform_color(image):
    """
    transforms 255-graylevel values to 0,1,2 as mask values
    """
    uniques = np.unique(image[-1])
    for idx,elem in enumerate(uniques):
        mask = np.where(image == elem)
        image[mask] = idx

        # one mask had four instead of three unique pixel values --> make sure the additional one is converted correctly
        mask2 = np.where(image == 178)
        image[mask2] = 1
    return image

def make_visible(image):
    """
    transforms mask values to gralevel values
    """
    image[image==0] = 0
    image[image==1] = 76
    image[image==2] = 255

    return image


def crop_center_square(image, im_size=480):
    """
    crop center of image
    """
    size=im_size
    # original image dimensions
    height, width = image.shape[:2]
    # calculate new dimensions
    new_width = new_height = size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    # crop
    cropped_image = image[top:bottom, left:right]
    return cropped_image


def resize_image(image, size=(640,480)):
    """
    resize image to specified size
    """
    # interpolation nearest neighbour to keep number of pixel unique values. 3 corresponds to INTER_AREA
    image = cv2.resize(image, dsize=size, interpolation=0)
    return image
    

def transform_imgs(image_path, save_path=None, im_size=(640,480)):
    """
    preprocessing for images: center crop
    """
    for idx, f in enumerate(os.listdir(image_path)):
        img = cv2.imread(os.path.join(image_path, f), 0)
        crp_img = crop_center_square(img, im_size)
        
        if save_path is not None:
            cv2.imwrite(os.path.join(save_path, '{}.png'.format(idx)), crp_img)
        return crp_img


def transform_masks(mask_path, save_path=None, im_size=(640,480)):
    """
    preprocessing for masks: transform color, resize, center crop
    """
    for idx, f in enumerate(os.listdir(mask_path)):
        img = cv2.imread(os.path.join(mask_path, f), 0)
        # first change color values (no black line), resize, crop
        col_img = transform_color(img)
        # original mask will have size (2345,3210)
        res_img = resize_image(col_img, im_size)
        crp_img = crop_center_square(res_img, im_size)

        if save_path is not None:
            cv2.imwrite(os.path.join(save_path, '{}.png'.format(idx)), crp_img)
        return crp_img