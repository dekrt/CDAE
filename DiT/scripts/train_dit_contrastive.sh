CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  train_contrastive.py \
  --model DiT-XL/2 \
  --data-path /lpai/dataset/imagenet-1k/0-1-0/train \
  --global-batch-size 384 \
  --epochs 30 \
  --results-dir /lpai/output/models \
  --freezing-decoder \
  --fixed-timesteps \
  --ckpt /lpai/inputs/models/ditssl-25-03-02-1/cdae/pretrained_models/DiT-XL-2-256x256.pt \
  # --ckpt /lpai/zxk/ddae/DiT/pretrained_models/DiT-XL-2-256x256.pt \
