#!/bin/bash

ckpts=(
  "/lpai/models/ditssl/cadedit100ep/000-DiT-XL-2/checkpoints/0050000.pt"
  "/lpai/models/ditssl/cadedit100ep/000-DiT-XL-2/checkpoints/0100000.pt"
  "/lpai/models/ditssl/cadedit100ep/000-DiT-XL-2/checkpoints/0150000.pt"
  "/lpai/models/ditssl/cadedit100ep/000-DiT-XL-2/checkpoints/0200000.pt"
  "/lpai/models/ditssl/cadedit100ep/000-DiT-XL-2/checkpoints/0250000.pt"
)

TRAIN_PATH="/lpai/zxk/ddae/DiT/dataset"
VAL_PATH="/lpai/zxk/ddae/DiT/dataset"
BATCH_SIZE=256

for ckpt in "${ckpts[@]}"
do
  echo "=============================="
  echo "Running with ckpt: $ckpt"
  echo "=============================="

  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    linear_probing.py \
    --dataset cifar \
    --train-data-path "$TRAIN_PATH" \
    --val-data-path "$VAL_PATH" \
    --batch-size "$BATCH_SIZE" \
    --ckpt "$ckpt"\
    --epoch 1 

  echo "Finished running ckpt: $ckpt"
  echo ""
done
