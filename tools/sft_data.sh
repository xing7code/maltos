#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PRESET=${PRESET:-olmo}
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
  PRESET=$1
  shift
fi

case "$PRESET" in
  olmo|olmo2|olmo2_13b_sft)
    # Keep preprocessing aligned with the current public Open Instruct 13B
    # recipe. It tokenizes from the base model and uses the updated -0225 mix.
    TOKENIZER_DEFAULT=allenai/OLMo-2-1124-13B
    DATASET_DEFAULT=allenai/tulu-3-sft-olmo-2-mixture-0225
    DATASET_CONFIG_DEFAULT=
    CHAT_TEMPLATE_FILE_DEFAULT=configs/chat_templates/olmo2_sft_masked.jinja
    SPLIT_DEFAULT=train
    MESSAGES_COLUMN_DEFAULT=messages
    PROMPT_COLUMN_DEFAULT=prompt
    COMPLETION_COLUMN_DEFAULT=completion
    OUTPUT_DIR_DEFAULT=workspace/olmo2_13b_sft/data/olmo2_13b_sft
    SEQ_LEN_DEFAULT=4096
    SEQUENCES_PER_SHARD_DEFAULT=8192
    MAX_EXAMPLES_DEFAULT=
    MAX_SEQUENCES_DEFAULT=
    EXPECTED_VOCAB_SIZE_DEFAULT=
    PACKING_ALGORITHM_DEFAULT=best_fit_decreasing
    PACKING_BUFFER_SIZE_DEFAULT=4096
    ASSISTANT_ONLY_LOSS_DEFAULT=true
    APPLY_CHAT_TEMPLATE_DEFAULT=true
    APPEND_EOS_DEFAULT=true
    STREAMING_DEFAULT=false
    SEED_DEFAULT=8
    SHUFFLE_BUFFER_SIZE_DEFAULT=10000
    LOG_EVERY_EXAMPLES_DEFAULT=1000
    ;;
  custom|none)
    TOKENIZER_DEFAULT=
    DATASET_DEFAULT=
    DATASET_CONFIG_DEFAULT=
    CHAT_TEMPLATE_FILE_DEFAULT=
    SPLIT_DEFAULT=train
    MESSAGES_COLUMN_DEFAULT=messages
    PROMPT_COLUMN_DEFAULT=prompt
    COMPLETION_COLUMN_DEFAULT=completion
    OUTPUT_DIR_DEFAULT=workspace/sft/data/sft
    SEQ_LEN_DEFAULT=4096
    SEQUENCES_PER_SHARD_DEFAULT=8192
    MAX_EXAMPLES_DEFAULT=
    MAX_SEQUENCES_DEFAULT=
    EXPECTED_VOCAB_SIZE_DEFAULT=
    PACKING_ALGORITHM_DEFAULT=best_fit_decreasing
    PACKING_BUFFER_SIZE_DEFAULT=4096
    ASSISTANT_ONLY_LOSS_DEFAULT=true
    APPLY_CHAT_TEMPLATE_DEFAULT=true
    APPEND_EOS_DEFAULT=true
    STREAMING_DEFAULT=false
    SEED_DEFAULT=42
    SHUFFLE_BUFFER_SIZE_DEFAULT=10000
    LOG_EVERY_EXAMPLES_DEFAULT=1000
    ;;
  *)
    echo "unknown PRESET=$PRESET; expected one of: olmo, olmo2, olmo2_13b_sft, custom, none" >&2
    exit 1
    ;;
esac

TOKENIZER=${TOKENIZER:-$TOKENIZER_DEFAULT}
DATASET=${DATASET:-$DATASET_DEFAULT}
DATASET_CONFIG=${DATASET_CONFIG:-$DATASET_CONFIG_DEFAULT}
CHAT_TEMPLATE_FILE=${CHAT_TEMPLATE_FILE:-$CHAT_TEMPLATE_FILE_DEFAULT}
SPLIT=${SPLIT:-$SPLIT_DEFAULT}
MESSAGES_COLUMN=${MESSAGES_COLUMN:-$MESSAGES_COLUMN_DEFAULT}
PROMPT_COLUMN=${PROMPT_COLUMN:-$PROMPT_COLUMN_DEFAULT}
COMPLETION_COLUMN=${COMPLETION_COLUMN:-$COMPLETION_COLUMN_DEFAULT}
OUTPUT_DIR=${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}
SEQ_LEN=${SEQ_LEN:-$SEQ_LEN_DEFAULT}
SEQUENCES_PER_SHARD=${SEQUENCES_PER_SHARD:-$SEQUENCES_PER_SHARD_DEFAULT}
MAX_EXAMPLES=${MAX_EXAMPLES:-$MAX_EXAMPLES_DEFAULT}
MAX_SEQUENCES=${MAX_SEQUENCES:-$MAX_SEQUENCES_DEFAULT}
EXPECTED_VOCAB_SIZE=${EXPECTED_VOCAB_SIZE:-$EXPECTED_VOCAB_SIZE_DEFAULT}
PACKING_ALGORITHM=${PACKING_ALGORITHM:-$PACKING_ALGORITHM_DEFAULT}
PACKING_BUFFER_SIZE=${PACKING_BUFFER_SIZE:-$PACKING_BUFFER_SIZE_DEFAULT}
ASSISTANT_ONLY_LOSS=${ASSISTANT_ONLY_LOSS:-$ASSISTANT_ONLY_LOSS_DEFAULT}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-$APPLY_CHAT_TEMPLATE_DEFAULT}
APPEND_EOS=${APPEND_EOS:-$APPEND_EOS_DEFAULT}
STREAMING=${STREAMING:-$STREAMING_DEFAULT}
SEED=${SEED:-$SEED_DEFAULT}
SHUFFLE_BUFFER_SIZE=${SHUFFLE_BUFFER_SIZE:-$SHUFFLE_BUFFER_SIZE_DEFAULT}
LOG_EVERY_EXAMPLES=${LOG_EVERY_EXAMPLES:-$LOG_EVERY_EXAMPLES_DEFAULT}

if [ -z "$TOKENIZER" ]; then
  echo "TOKENIZER must be set when PRESET=$PRESET" >&2
  exit 1
fi
if [ -z "$DATASET" ]; then
  echo "DATASET must be set when PRESET=$PRESET" >&2
  exit 1
fi

echo "preset=$PRESET tokenizer=$TOKENIZER dataset=$DATASET output_dir=$OUTPUT_DIR"

set -- \
  python3 \
  tools/prepare_sft_shards.py \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --messages-column "$MESSAGES_COLUMN" \
  --prompt-column "$PROMPT_COLUMN" \
  --completion-column "$COMPLETION_COLUMN" \
  --tokenizer-name-or-path "$TOKENIZER" \
  --output-dir "$OUTPUT_DIR" \
  --seq-len "$SEQ_LEN" \
  --sequences-per-shard "$SEQUENCES_PER_SHARD" \
  --packing-algorithm "$PACKING_ALGORITHM" \
  --packing-buffer-size "$PACKING_BUFFER_SIZE" \
  --seed "$SEED" \
  --shuffle-buffer-size "$SHUFFLE_BUFFER_SIZE" \
  --log-every-examples "$LOG_EVERY_EXAMPLES"

if [ -n "$DATASET_CONFIG" ]; then
  set -- "$@" --config "$DATASET_CONFIG"
fi
if [ -n "$CHAT_TEMPLATE_FILE" ]; then
  set -- "$@" --chat-template-file "$CHAT_TEMPLATE_FILE"
fi
if [ -n "$MAX_EXAMPLES" ]; then
  set -- "$@" --max-examples "$MAX_EXAMPLES"
fi
if [ -n "$MAX_SEQUENCES" ]; then
  set -- "$@" --max-sequences "$MAX_SEQUENCES"
fi
if [ -n "$EXPECTED_VOCAB_SIZE" ]; then
  set -- "$@" --expected-vocab-size "$EXPECTED_VOCAB_SIZE"
fi
if [ "$ASSISTANT_ONLY_LOSS" = "true" ]; then
  set -- "$@" --assistant-only-loss
else
  set -- "$@" --no-assistant-only-loss
fi
if [ "$APPLY_CHAT_TEMPLATE" = "true" ]; then
  set -- "$@" --apply-chat-template
else
  set -- "$@" --no-apply-chat-template
fi
if [ "$APPEND_EOS" = "true" ]; then
  set -- "$@" --append-eos
else
  set -- "$@" --no-append-eos
fi
if [ "$STREAMING" = "true" ]; then
  set -- "$@" --streaming
fi

"$@"
