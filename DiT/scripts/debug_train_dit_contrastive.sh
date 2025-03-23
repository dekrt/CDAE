CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  train_contrastive.py \
  --model DiT-XL/2 \
  --data-path /lpai/dataset/imagenet-1k/0-1-0/train \
  --global-batch-size 192 \
  --epochs 30 \
  --results-dir /lpai/output/models \
  --freezing-decoder \
  --lambda-lr \
  --log-every 1 \
  --ckpt /lpai/zxk/ddae/DiT/pretrained_models/DiT-XL-2-256x256.pt \
  # --modified-timesteps \