import sys
import os

# add parent directory to system path to be able to assess functions from root
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


for idx, file in enumerate(os.listdir('data/prediction/predicted/att_unet/grayscale/')):

    path_1 = os.path.join('data/prediction/predicted/unet/grayscale/{}.png'.format(idx))
    path_2 = os.path.join('data/prediction/predicted/att_unet/grayscale/{}.png'.format(idx))
    path_3 = os.path.join('data/prediction/predicted/psp_net/grayscale/{}.png'.format(idx))
    path_4 = os.path.join('data/prediction/predicted/unet_flight9/grayscale/{}.png'.format(idx))
    path_5 = os.path.join('data/prediction/predicted/att_unet_flight9/grayscale/{}.png'.format(idx))
    path_6 = os.path.join('data/prediction/predicted/psp_net_flight9/grayscale/{}.png'.format(idx))

    print(path_1)

    fig, ax = plt.subplots(2, 3)

    ax[0, 0].imshow(cv2.imread(path_1))
    ax[0, 1].imshow(cv2.imread(path_2))
    ax[0, 2].imshow(cv2.imread(path_3))
    ax[1, 0].imshow(cv2.imread(path_4))
    ax[1, 1].imshow(cv2.imread(path_5))
    ax[1, 2].imshow(cv2.imread(path_6))

    plt.savefig('comparison/models_flight9/{}.png'.format(idx))


"""
for i in range(10):
    im = ims[i]
    plt.imshow(im)
    plt.imsave('.im{}.png'.format(i), im)

for i in range(10):
    ma = mas[i]
    plt.imshow(im)
    plt.imsave('.ma{}.png'.format(i), ma)
"""