# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 contrastive.py --dataset cifar --use_amp
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 contrastive.py --dataset cifar --use_amp
CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 contrastive.py --dataset cifar --use_amp

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 contrastive.py --dataset cifar --use_amp
