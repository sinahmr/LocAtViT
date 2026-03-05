#!/bin/bash

MODEL=$1
DATASET=$2

export TIMM_FUSED_ATTN=2
export OMP_NUM_THREADS=8
DATASET=${DATASET} CHECKPOINT=${MODEL} python -m evaluations.segmentation.train --config "evaluations/segmentation/configs/simple.py"
