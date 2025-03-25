#!/bin/bash

ckpts=(
  "/lpai/inputs/models/ditssl-25-03-02-1/cdae/pretrained_models/DiT-XL-2-256x256.pt"
  "/lpai/inputs/models/ditssl-v2/final.pt"
)

CIFAR_TRAIN_PATH="/lpai/dataset/cifar10-lhp/0-1-0/CIFAR10"
CIFAR_VAL_PATH="/lpai/dataset/cifar10-lhp/0-1-0/CIFAR10"
TINY_TRAIN_PATH="/lpai/dataset/tinyimagenet/0-1-0/tiny-imagenet-200"
TINY_VAL_PATH="/lpai/dataset/tinyimagenet/0-1-0/tiny-imagenet-200"
RESULTS_PATH="/lpai/output/models"
BATCH_SIZE=384

for ckpt in "${ckpts[@]}"
do
  echo "=============================="
  echo "Running with ckpt: $ckpt"
  echo "=============================="

  echo "Running CIFAR"

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    linear_probing.py \
    --dataset cifar \
    --train-data-path "$CIFAR_TRAIN_PATH" \
    --val-data-path "$CIFAR_VAL_PATH" \
    --batch-size "$BATCH_SIZE" \
    --results-dir "$RESULTS_PATH" \
    --ckpt "$ckpt"

  echo "Finished running CIFAR"
  echo ""

  echo "Running TINY"

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    linear_probing.py \
    --dataset tiny \
    --train-data-path "$TINY_TRAIN_PATH" \
    --val-data-path "$TINY_VAL_PATH" \
    --batch-size "$BATCH_SIZE" \
    --results-dir "$RESULTS_PATH" \
    --ckpt "$ckpt"

  echo "Finished running TINY"
  echo ""

  echo "=============================="
  echo "Finished running ckpt: $ckpt"
  echo "=============================="

done