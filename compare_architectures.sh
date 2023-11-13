#!/bin/bash

echo "Investigating if attention is worth it..."

python fine_tune.py --pref test_att_unet --path_to_config config/att_unet.json
python fine_tune.py --pref test_unet --path_to_config config/best_unet.json

echo "Model selection done. Logs are stored in 'metrics/scores/fine_tune/'."