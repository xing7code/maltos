TOKENIZER=${TOKENIZER:-Qwen/Qwen2.5-7B}
DATASET=${DATASET:-HuggingFaceFW/fineweb-edu}
DATASET_CONFIG=${DATASET_CONFIG:-sample-10BT}
SPLIT=${SPLIT:-train}
TEXT_COLUMN=${TEXT_COLUMN:-text}
OUTPUT_DIR=${OUTPUT_DIR:-datasets/fineweb_500m}
MAX_TOKENS=${MAX_TOKENS:-500000000}
TOKENS_PER_SHARD=${TOKENS_PER_SHARD:-100000000}

python3 tools/prepare_token_shards.py \
  --dataset "$DATASET" \
  --config "$DATASET_CONFIG" \
  --split "$SPLIT" \
  --column "$TEXT_COLUMN" \
  --tokenizer-name-or-path "$TOKENIZER" \
  --output-dir "$OUTPUT_DIR" \
  --max-tokens "$MAX_TOKENS" \
  --tokens-per-shard "$TOKENS_PER_SHARD" \
  --streaming
