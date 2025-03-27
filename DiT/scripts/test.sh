#!/bin/bash

ckpts=(
  "/lpai/zxk/ddae/DiT/pretrained_models/DiT-XL-2-256x256.pt"
  "/lpai/models/ditssl/v2/final.pt"
)

for ckpt in "${ckpts[@]}"
do
  echo "=============================="
  echo "Running with ckpt: $ckpt"
  echo "=============================="

  CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  linear_probing.py \
  --dataset cifar \
  --train-data-path /lpai/zxk/ddae/DiT/dataset \
  --val-data-path /lpai/zxk/ddae/DiT/dataset \
  --batch-size 128 \
  --results-dir /lpai/output/models \
  --ckpt "$ckpt" \
  --use-amp \

  echo "Finished running ckpt: $ckpt"
  echo ""
done
