CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  train_contrastive.py \
  --model DiT-XL/2 \
  --num-classes 10 \
  --global-batch-size 256 \
  --epochs 600 \
  --results-dir /lpai/output/models
