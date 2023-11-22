import sys
import os

# add parent directory to system path to be able to assess functions from root
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2

parser = argparse.ArgumentParser(description="Plots preselected IR prdictions with corresponding ground truth.")

# prefix
parser.add_argument("--date", default="220719_2", type=str, help="Date ID.")

def main():
    args = parser.parse_args()
    params = vars(args)

    for idx, file in enumerate(os.listdir(os.path.join('data/selected/', params['date']))):

        num = int(file.split('.')[0])

        path_1 = os.path.join('data/selected/', params['date'], file)
        path_2 = os.path.join('data/prediction/preprocessed/{0}/{1}.png'.format(params['date'], num*4))

        print(path_1)
        print(path_2)

        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(cv2.imread(path_1))
        ax[1].imshow(cv2.imread(path_2))

        plt.savefig('comparison/to_ir/{0}/{1}'.format(params['date'], file))

if __name__ == "__main__":
    main()