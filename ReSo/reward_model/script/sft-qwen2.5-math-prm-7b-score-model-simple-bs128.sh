# This script should be run on four A100-80G GPUs within a single node.

export TOKENIZERS_PARALLELISM=false

accelerate launch train.py \
    --base_model "/path/to/your/model" \
    --data_train "./dataset/score_model_simple_train.jsonl" \
    --data_test "./dataset/score_model_simple_test.jsonl" \
    --run_name "sft-qwen2.5-math-prm-7b-score-model-simple-bs128" \
    --save_path "/path/to/your/checkpoint" \
    --logging_path "/path/to/your/log" \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --context_length 4096 \
    --batch_size_train 2 \
    --batch_size_eval 16 \
    --grad_accumulation 16 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --num_epochs 5 \
    --logging_steps 1 \
    --eval_steps 10 \
    --num_workers 4

python test.py \
    --base_model "/path/to/base/model" \
    --lora_model "/path/to/lora/model" \
    --data_path "./dataset/score_model_simple_test.jsonl" \
    --device "cuda:0"
