#!/bin/bash

echo "Flight 3"

#python predict.py --pref flight_3_semi --data 'data/prediction/raw/220713_1/IRdata_ATWAICE_processed_220713_075354.nc' --weights_path weights/att_unet/best_modelsemi_01.h5  --convert_to_grayscale

echo "Flight 4"

#python predict.py --pref flight_4_semi --data 'data/prediction/raw/220713_2/IRdata_ATWAICE_processed_220713_104532.nc' --weights_path weights/att_unet/best_modelsemi_01.h5  --convert_to_grayscale

echo "Flight 6"

#python predict.py --pref flight_6_semi --data 'data/prediction/raw/220717_1/IRdata_ATWAICE_processed_220717_075915.nc' --weights_path weights/att_unet/best_modelsemi_01.h5  --convert_to_grayscale

echo "Flight 7"

python predict.py --pref flight_7_semi --data 'data/prediction/raw/220717_2/IRdata_ATWAICE_processed_220717_122355.nc' --weights_path weights/att_unet/best_modelsemi_01.h5  --convert_to_grayscale

echo "Flight 8"

#python predict.py --pref flight_8_semi --data 'data/prediction/raw/220718_1/IRdata_ATWAICE_processed_220718_081257.nc' --weights_path weights/att_unet/best_modelsemi_01.h5  --convert_to_grayscale

echo "Flight 9"

python predict.py --pref flight_9_semi --data 'data/prediction/raw/220718_2/IRdata_ATWAICE_processed_220718_142920.nc' --weights_path weights/att_unet/best_modelsemi_01.h5  --convert_to_grayscale

echo "Flight 10"

python predict.py --pref flight_10_semi --data 'data/prediction/raw/220719_1/IRdata_ATWAICE_processed_220719_104906.nc' --weights_path weights/att_unet/best_modelsemi_01.h5  --convert_to_grayscale

echo "Flight 11"

python predict.py --pref flight_11_semi --data 'data/prediction/raw/220719_2/IRdata_ATWAICE_processed_220719_112046.nc' --weights_path weights/att_unet/best_modelsemi_01.h5  --convert_to_grayscale

echo "Flight 13"

#python predict.py --pref flight_13_semi --data 'data/prediction/raw/220724/IRdata_ATWAICE_processed_220724_131311.nc' --weights_path weights/att_unet/best_modelsemi_01.h5  --convert_to_grayscale

echo "Flight 14"

#python predict.py --pref flight_14_semi --data 'data/prediction/raw/220730_1/IRdata_ATWAICE_processed_220730_042841.nc' --weights_path weights/att_unet/best_modelsemi_01.h5  --convert_to_grayscale

echo "Flight 15"

#python predict.py --pref flight_15_semi --data 'data/prediction/raw/220730_2/IRdata_ATWAICE_processed_220730_085252.nc' --weights_path weights/att_unet/best_modelsemi_01.h5 --convert_to_grayscale

echo "Flight 16"

python predict.py --pref flight_16_semi --data 'data/prediction/raw/220730_3/IRdata_ATWAICE_processed_220730_111439.nc' --weights_path weights/att_unet/best_modelsemi_01.h5  --convert_to_grayscale

echo "Flight 17"

#python predict.py --pref flight_17_semi --data 'data/prediction/raw/220808/IRdata_ATWAICE_processed_220808_084908.nc' --weights_path weights/att_unet/best_modelsemi_01.h5  --convert_to_grayscale

