#!/bin/bash

ckpts=(
  "/lpai/inputs/models/ditssl-25-03-02-1/cdae/pretrained_models/DiT-XL-2-256x256.pt"
  "/lpai/inputs/models/ditssl-cadedit100ep/000-DiT-XL-2/checkpoints/0050000.pt"
  "/lpai/inputs/models/ditssl-cadedit100ep/000-DiT-XL-2/checkpoints/0100000.pt"
  "/lpai/inputs/models/ditssl-cadedit100ep/000-DiT-XL-2/checkpoints/0150000.pt"
  "/lpai/inputs/models/ditssl-cadedit100ep/000-DiT-XL-2/checkpoints/0200000.pt"
  "/lpai/inputs/models/ditssl-cadedit100ep/000-DiT-XL-2/checkpoints/0250000.pt"
)

TRAIN_PATH="/lpai/dataset/cifar10-lhp/0-1-0/CIFAR10"
VAL_PATH="/lpai/dataset/cifar10-lhp/0-1-0/CIFAR10"
BATCH_SIZE=256

for ckpt in "${ckpts[@]}"
do
  echo "=============================="
  echo "Running with ckpt: $ckpt"
  echo "=============================="

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    linear_probing.py \
    --dataset cifar \
    --train-data-path "$TRAIN_PATH" \
    --val-data-path "$VAL_PATH" \
    --batch-size "$BATCH_SIZE" \
    --ckpt "$ckpt"

  echo "Finished running ckpt: $ckpt"
  echo ""
done
