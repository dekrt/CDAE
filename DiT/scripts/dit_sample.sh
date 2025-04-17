ckpts=(
    # "/lpai/inputs/models/ditssl-25-03-02-1/cdae/pretrained_models/DiT-XL-2-256x256.pt"
    "/lpai/inputs/models/ditssl-v2/final.pt"
    # "/lpai/inputs/models/ditssl-simclr/final.pt"
    # "/lpai/inputs/models/ditssl-vicreg/final.pt"
)

for ckpt in "${ckpts[@]}"
do
    echo "=============================="
    echo "Running with ckpt: $ckpt"
    echo "=============================="
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    sample_ddp.py \
    --model DiT-XL/2 \
    --num-fid-samples 50000 \
    --image-size 256 \
    --ckpt "$ckpt" \
    --sample-dir "/lpai/output/models/samples"
    echo "Finished running ckpt: $ckpt"
    echo ""
done