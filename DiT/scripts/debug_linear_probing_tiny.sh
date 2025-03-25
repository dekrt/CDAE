CUDA_VISIBLE_DEVICES=6,7 torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  linear_probing.py \
  --dataset tiny \
  --train-data-path /lpai/zxk/ddae/data/tiny-imagenet-200 \
  --val-data-path /lpai/zxk/ddae/data/tiny-imagenet-200 \
  --batch-size 384 \
  --ckpt /lpai/zxk/ddae/DiT/pretrained_models/DiT-XL-2-256x256.pt \
  --results-dir /lpai/output/models \
  # --epoch 1 \
  # --ckpt /lpai/models/ditssl/cadedit100ep/000-DiT-XL-2/checkpoints/0200000.pt \
  # --ckpt /lpai/inputs/models/ditssl-25-03-02-1/cdae/pretrained_models/DiT-XL-2-256x256.pt \
  # --ckpt /lpai/output/models/results/000-DiT-XL-2/checkpoints/0300000.pt \
