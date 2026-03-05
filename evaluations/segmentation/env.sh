#!/bin/bash

conda create -n seg python==3.10
conda activate seg

pip install -r evaluations/segmentation/seg_requirements.txt
pip install -U openmim -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
mim install mmcv-full==1.7.2
pip install numpy==1.26.4
pip install mmsegmentation==0.30.0

echo "Perform these steps to make sure mmcv works:"
echo "    From your env folder, open '{env}/lib/python3.10/site-packages/mmcv/parallel/_functions.py'"
echo '    and replace line 75 (streams = [_get_stream(device) for device in target_gpus]) to the following:'
echo '    streams = [_get_stream(torch.device("cuda", device)) for device in target_gpus]'
