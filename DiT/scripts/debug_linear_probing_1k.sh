CUDA_VISIBLE_DEVICES=6,7 torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  linear_probing.py \
  --train-data-path /lpai/dataset/imagenet-1k/0-1-0/train \
  --val-data-path /lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val \
  --dataset imagenet \
  --batch-size 32 \
  --ckpt /lpai/zxk/ddae/DiT/pretrained_models/DiT-XL-2-256x256.pt \
  # --ckpt /lpai/inputs/models/ditssl-25-03-02-1/cdae/pretrained_models/DiT-XL-2-256x256.pt \
  # --ckpt /lpai/output/models/results/000-DiT-XL-2/checkpoints/0300000.pt \
  # --use-amp \
