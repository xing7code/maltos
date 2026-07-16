from __future__ import annotations

IGNORE_INDEX = -100
PAD_SEQUENCE_ID = -1

INPUT_IDS_KEY = "input_ids"
LABELS_KEY = "labels"
POSITION_IDS_KEY = "position_ids"
SEQUENCE_IDS_KEY = "sequence_ids"
POSITION_OFFSET_KEY = "position_offset"
LOSS_WEIGHT_KEY = "loss_weight"
HIDDEN_STATES_KEY = "hidden_states"

MODEL_BATCH_KEYS = (
    INPUT_IDS_KEY,
    LABELS_KEY,
    POSITION_IDS_KEY,
    SEQUENCE_IDS_KEY,
    POSITION_OFFSET_KEY,
    LOSS_WEIGHT_KEY,
    HIDDEN_STATES_KEY,
)

SFT_BATCH_KEYS = (
    INPUT_IDS_KEY,
    LABELS_KEY,
    POSITION_IDS_KEY,
    SEQUENCE_IDS_KEY,
)
