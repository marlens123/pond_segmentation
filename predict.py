import os
import re
import cv2
import csv
import netCDF4
import argparse
import matplotlib.pyplot as plt
from utils.predict_helpers import calculate_mpf, predict_image, crop_center_square

parser = argparse.ArgumentParser(description="Uses trained model to predict and store surface masks from netCDF file containing TIR images from a single helicopter flight. Optional calculation of melt pond fraction (MPF).")

parser.add_argument("--data", type=str, help="Either: 1) Filename of netCDF data file. For this, data must be stored in 'data/prediction/raw'. Or: 2) Absolute path to netCDF data file. Then data must not be copied in advance.")
parser.add_argument("--weights_path", default="weights/flight9_flight16.h5", type=str, help="Path to model weights that should be used.")
parser.add_argument("--preprocessed_path", default="data/prediction/preprocessed", type=str, help="Path to folder that should store the preprocessed images.")
parser.add_argument("--predicted_path", default="data/prediction/predicted", type=str, help="Path to folder that should store the predicted image masks.")
parser.add_argument("--metrics_path", default="metrics/melt_pond_fraction/mpf.csv", type=str, help="Path to .csv file that should store the resulting mpf (if calculation is desired).")
parser.add_argument("--skip_mpf", action="store_true", help="Skips the calculation of the melt pond fraction for the predicted flight.")
parser.add_argument("--skip_prediction", action="store_true", help="Skips prediction process. Can be used to directly perform mpf calculation. In that case, 'predicted_path' must contain predicted images.")

def main():
    args = parser.parse_args()
    params = vars(args)

    # extract date of flight used
    match = re.search(r"(\d{6})_(\d{6})", params['data'])

    if match:
        date_part = match.group(1)
        time_part = match.group(2)

        # formatting the date
        formatted_date = f"20{date_part[:2]}-{date_part[2:4]}-{date_part[4:]}"
        print(f"The date in the filename is: {formatted_date}")
    else:
        print("Date not found in the filename.")

    if not params['skip_prediction']:

        # load data and store as images
        # use whole path when abs path is given, else use data from 'data/prediction/raw'
        if '/' in params['data']:
            ds = netCDF4.Dataset(params['data'])
            print("Abs path is used.")
        else:
            ds = netCDF4.Dataset(os.path.join('data/prediction/raw', params['data']))
            print("Rel path is used.")
        imgs = ds.variables['Ts'][:]

        tmp = []

        for im in imgs:
            im = crop_center_square(im)
            tmp.append(im)

        imgs = tmp

        print("Start extracting images...")

        # extract only every 4th image to avoid overlap
        for idx, img in enumerate(imgs):
            if(idx % 4 == 0):
                plt.imsave(os.path.join(params['preprocessed_path'], '{}.png'.format(idx)), img, cmap='gray')

        print("Start predicting images...")

        # extract surface masks from images
        for idx, file in enumerate(os.listdir(params['preprocessed_path'])):
            if file.endswith('.png'):
                img = cv2.imread(os.path.join(params['preprocessed_path'], file), 0)
                predict_image(img, 480, params['weights_path'], backbone='resnet34', train_transfer='imagenet', save_path=os.path.join(params['predicted_path'],'{}.png'.format(idx)), visualize=False)

    # optionally calculate melt pond fraction and store in csv file
    if not params['skip_mpf']:
        mpf = calculate_mpf(params['predicted_path'])

        headers = ['flight_date', 'melt_pond_fraction']

        with open(params['metrics_path'], 'a', newline='') as f:
            writer = csv.writer(f)

            # headers in the first row
            if f.tell() == 0:
                writer.writerow(headers)

            writer.writerow([formatted_date, mpf])

    print("Process ended.")

if __name__ == "__main__":
    main()

    



