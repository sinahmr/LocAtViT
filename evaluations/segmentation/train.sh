#!/bin/bash

MODEL=$1
DATASET=$2

export TIMM_FUSED_ATTN=2
DATASET=${DATASET} CHECKPOINT=${MODEL} TIMM_FUSED_ATTN=2 OMP_NUM_THREADS=8 \
    python -m evaluations.segmentation.train --config "evaluations/segmentation/configs/simple.py"
