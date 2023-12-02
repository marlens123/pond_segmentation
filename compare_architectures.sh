#!/bin/bash

#echo "Investigating if attention is worth it..."

#python fine_tune.py --pref test_att_unet --path_to_config config/att_unet.json
#python fine_tune.py --pref test_unet --path_to_config config/best_unet.json

#echo "Model selection done. Logs are stored in 'metrics/scores/fine_tune/'."

echo "Investigating if pspnet is worth it..."

#python fine_tune.py --pref test_psp_net --path_to_config config/psp_net.json
python predict.py --pref unet --data IRdata_ATWAICE_processed_220718_142920.nc --weights_path weights/unet/best_modeltest_unet.h5 --convert_to_grayscale --skip_preprocessing

echo "Done."