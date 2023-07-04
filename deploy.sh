
pip install -r requirements.txt

cp finetune-alpaca/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cuda117.so finetune-alpaca/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so
echo "bnb cuda library modified"

python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights ./scroltest-rUgkyuiEUWYTz8ileHSArxtxekC3
