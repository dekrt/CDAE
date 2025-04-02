CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  train_DiT_Classifier.py \
  --train-data-path /lpai/dataset/imagenet-1k/0-1-0/train \
  --val-data-path /lpai/dataset/imagenet-1k/0-1-0/ILSVRC2012/val \
  --batch-size 128 \
  --result-dir /lpai/output/models \