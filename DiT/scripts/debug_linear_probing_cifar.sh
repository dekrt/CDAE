CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  linear_probing.py \
  --dataset cifar \
  --train-data-path /lpai/zxk/ddae/DiT/dataset \
  --val-data-path /lpai/zxk/ddae/DiT/dataset \
  --batch-size 256 \
  --ckpt /lpai/zxk/ddae/DiT/pretrained_models/DiT-XL-2-256x256.pt \
  # --ckpt /lpai/models/ditssl/cadedit100ep/000-DiT-XL-2/checkpoints/0200000.pt \
  # --ckpt /lpai/inputs/models/ditssl-25-03-02-1/cdae/pretrained_models/DiT-XL-2-256x256.pt \
  # --ckpt /lpai/output/models/results/000-DiT-XL-2/checkpoints/0300000.pt \
  # --use-amp \
