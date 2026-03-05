#!/bin/bash

export TIMM_FUSED_ATTN=2
PORT=$(comm -23 <(seq 29401 30401 | sort) <(ss -tuln | awk '{print $5}' | grep -oE '[0-9]+$' | sort | uniq) | shuf -n 1)

python -m torch.distributed.launch --nproc_per_node=1 --master_port "$PORT" eval_knn.py \
  --num_workers 16 --batch_size_per_gpu 1024 "$@"
