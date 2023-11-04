import os
import cv2
import netCDF4
import argparse
import matplotlib.pyplot as plt
from utils.predict_helpers import calculate_mpf, predict_image, crop_center_square


### To-Do: Store the resulting MPFs in a CSV file
### retrieve the index / identifier of the flight
### include disclaimer that much storage is needed

parser = argparse.ArgumentParser(description="Uses trained model to predict and store surface masks from netCDF file containing TIR images from a single helicopter flight. Optional calculation of melt pond fraction (MPF).")

parser.add_argument("--data", type=str, help="Path to netCDF data file.")
parser.add_argument("--weights", default="weights/flight9_flight16.h5", type=str, help="Path to model weights that should be used.")
parser.add_argument("--preprocessed_path", type=str, help="Path to folder that should store the preprocessed images.")
parser.add_argument("--predicted_path", type=str, help="Path to folder that should store the predicted image masks.")
parser.add_argument("--mpf", action='store_false', help="Whether to return the melt pond fraction for the predicted flight.")

def main():
    args = parser.parse_args()
    params = vars(args)

    # load data and store as images
    ds = netCDF4.Dataset(params['data'])
    imgs = ds.variables['Ts'][:]

    tmp = []

    for im in imgs:
        im = crop_center_square(im)
        tmp.append(im)

    imgs = tmp

    for idx, img in enumerate(imgs):
        plt.imsave(os.path.join(params['preprocessed_path'], '{}.png'.format(idx)), img, cmap='gray')

    # process surface masks from images
    for idx, file in enumerate(os.listdir(params['preprocessed_path'])):
        img = cv2.imread(os.path.join(params['preprocessed_path'], file), 0)
        predict_image(img, 480, params['weights'], backbone='resnet34', train_transfer='imagenet', save_path=os.path.join(params['predicted_path'],'{}.png'.format(idx)), visualize=False)

    # optionally calculate melt pond fraction and store in csv file
    if params['mpf']:
        mpf = calculate_mpf(params['predicted_path'])


if __name__ == "__main__":
    main()

    



