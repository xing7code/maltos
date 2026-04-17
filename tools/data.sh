TOKENIZER=${TOKENIZER:-Qwen/Qwen2.5-7B}
DATASET=${DATASET:-HuggingFaceFW/fineweb-edu}
DATASET_CONFIG=${DATASET_CONFIG:-sample-10BT}
SPLIT=${SPLIT:-train}
TEXT_COLUMN=${TEXT_COLUMN:-text}

python3 tools/prepare_data.py \
  --tokenizer-name-or-path "$TOKENIZER" \
  --output-folder datasets/fineweb_500m \
  --n-tasks 16 \
  --target-tokens 500000000 \
  --estimated-total-tokens 10000000000 \
  hf --dataset "$DATASET" --config "$DATASET_CONFIG" --split "$SPLIT" --column "$TEXT_COLUMN"
