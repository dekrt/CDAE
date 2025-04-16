#!/bin/bash

real_npz="/lpai/inputs/models/ditssl-virtual/VIRTUAL_imagenet256_labeled.npz"
sample_dir="/lpai/output/models/samples"
log_file="/lpai/output/models/samples/evaluation_results.txt"

> "$log_file"

for npz_file in "$sample_dir"/*.npz
do
    tag_name=$(basename "$npz_file" .npz)

    echo "=============================="
    echo "Evaluating: $npz_file"
    echo "=============================="

    echo ">>> [BEGIN EVAL] $tag_name" >> "$log_file"
    python evaluator.py "$real_npz" "$npz_file" >> "$log_file" 2>&1
    echo "<<< [END EVAL] $tag_name" >> "$log_file"
    echo "" >> "$log_file"

    echo "" 
done
