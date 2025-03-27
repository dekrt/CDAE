CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  linear_probing.py \
  --dataset cifar \
  --train-data-path /lpai/zxk/ddae/DiT/dataset \
  --val-data-path /lpai/zxk/ddae/DiT/dataset \
  --batch-size 128 \
  --results-dir /lpai/output/models \
  --ckpt /lpai/zxk/ddae/DiT/pretrained_models/DiT-XL-2-256x256.pt \
  --use-amp \
  # --epoch 1 \
  # --ckpt /lpai/models/ditssl/v2/final.pt
  # --ckpt /lpai/models/ditssl/cadedit100ep/000-DiT-XL-2/checkpoints/0200000.pt \
  # --ckpt /lpai/inputs/models/ditssl-25-03-02-1/cdae/pretrained_models/DiT-XL-2-256x256.pt \
  # --ckpt /lpai/output/models/results/000-DiT-XL-2/checkpoints/0300000.pt \
