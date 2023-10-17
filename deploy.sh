lora_weights_dir=$1
base_model=$2
python3 -m venv "finetune-alpaca"

source "finetune-alpaca/bin/activate" 
echo "venv activated"

pip install -r requirements.txt

cp finetune-alpaca/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cuda117.so finetune-alpaca/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so
echo "bnb cuda library modified"

python inference.py \
    --base_model $base_model \
    --lora_weights_urls $lora_weights_urls \
