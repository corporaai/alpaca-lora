#!/bin/bash
dataset_url=$3
model_name=$1
model_card=$2
scrol_token=$4
jobid=$5

python3 -m venv "finetune-alpaca"

source "finetune-alpaca/bin/activate" 
echo "venv activated"
pip install -r requirements.txt

cp finetune-alpaca/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cuda117.so finetune-alpaca/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so
echo "bnb cuda library modified"

python train.py \
    --model_name $1 \
    --model_card $2 \
    --dataset_url $3 \
    --scrol_token $4 \
    --jobid $5