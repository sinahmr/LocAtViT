#!/bin/bash

NUM_PROC=$1
shift 1

export TIMM_FUSED_ATTN=2
PORT=$(comm -23 <(seq 29401 30401 | sort) <(ss -tuln | awk '{print $5}' | grep -oE '[0-9]+$' | sort | uniq) | shuf -n 1)

python -m torch.distributed.launch --nproc_per_node="$NUM_PROC" --master_port "$PORT" main_dino.py \
  --num_workers 8 --saveckp_freq 10 "$@"
