#!/usr/bin/env bash

if [ ! -d "data" ]; then
    # echo "Creating data/ directory..."
    mkdir -p data
fi

input_file="migration_others_dataset_cutoff_test_512.parquet"
output_file="migration_others_dataset_cutoff_test_512_output.parquet"
dataset_id="blackwhite1337/zTrans_dataset_512"
split="test"

model="deepseek-ai/deepseek-coder-6.7b-instruct"
device="0"
batch_size=5

max_length=1024
# max_new_tokens=1024
max_new_tokens=512
do_sample=false
top_k=50
top_p="0.95"

if [ "$do_sample" = true ]; then
    do_sample="--do_sample"
else
    do_sample=""
fi

time python phat_run_pipeline.py \
    --input_file=$input_file \
    --output_file=$output_file \
    --dataset_id=$dataset_id \
    --split=$split \
    --model=$model \
    --device=$device \
    --batch_size=$batch_size \
    --max_length=$max_length \
    --max_new_tokens=$max_new_tokens \
    $do_sample \
    --top_k=$top_k \
    --top_p=$top_p