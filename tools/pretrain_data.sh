#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TOKENIZER=${TOKENIZER:-NousResearch/Llama-2-7b-hf}
DATASET=${DATASET:-HuggingFaceFW/fineweb-edu}
DATASET_CONFIG=${DATASET_CONFIG:-sample-10BT}
SPLIT=${SPLIT:-train}
TEXT_COLUMN=${TEXT_COLUMN:-text}
OUTPUT_DIR=${OUTPUT_DIR:-datasets/fineweb_500m}
MAX_TOKENS=${MAX_TOKENS:-500000000}
TOKENS_PER_SHARD=${TOKENS_PER_SHARD:-100000000}
EXPECTED_VOCAB_SIZE=${EXPECTED_VOCAB_SIZE:-32000}

set -- \
  python3 \
  tools/prepare_token_shards.py \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --column "$TEXT_COLUMN" \
  --tokenizer-name-or-path "$TOKENIZER" \
  --expected-vocab-size "$EXPECTED_VOCAB_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  --max-tokens "$MAX_TOKENS" \
  --tokens-per-shard "$TOKENS_PER_SHARD" \
  --streaming

if [ -n "$DATASET_CONFIG" ]; then
  set -- "$@" --config "$DATASET_CONFIG"
fi

"$@"
