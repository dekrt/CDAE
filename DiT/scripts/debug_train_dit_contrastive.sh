CUDA_VISIBLE_DEVICES=1,5 torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  train_contrastive.py \
  --model DiT-XL/2 \
  --data-path /lpai/dataset/imagenet-1k/0-1-0/train \
  --global-batch-size 32 \
  --epochs 30 \
  --ckpt-every 25000 \
  --results-dir /lpai/output/models \
  --freezing-decoder \
  --lambda-lr \
  --log-every 10 \
  --loss-type vicreg \
  --ckpt /lpai/zxk/ddae/DiT/pretrained_models/DiT-XL-2-256x256.pt \
  # --modified-timesteps \