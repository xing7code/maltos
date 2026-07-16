from __future__ import annotations

import torch

from utils.constants import IGNORE_INDEX, INPUT_IDS_KEY, LABELS_KEY, POSITION_IDS_KEY, SEQUENCE_IDS_KEY


def causal_lm_batch(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = IGNORE_INDEX
    return input_ids, labels


def packed_causal_lm_batch(input_ids: torch.Tensor, *, num_sequences: int = 2) -> dict[str, torch.Tensor]:
    if input_ids.ndim != 2:
        raise ValueError("packed_causal_lm_batch expects [batch, seq] input_ids")
    batch_size, seq_len = input_ids.shape
    if num_sequences < 2:
        raise ValueError("packed_causal_lm_batch expects num_sequences >= 2")
    if num_sequences > seq_len:
        raise ValueError("packed_causal_lm_batch expects num_sequences <= seq_len")

    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = IGNORE_INDEX
    position_ids = torch.empty_like(input_ids)
    sequence_ids = torch.empty_like(input_ids)
    row_sequence_bases = torch.arange(batch_size, device=input_ids.device, dtype=torch.long).unsqueeze(1) * num_sequences

    start = 0
    base_length = seq_len // num_sequences
    remainder = seq_len % num_sequences
    for packed_idx in range(num_sequences):
        length = base_length + int(packed_idx < remainder)
        end = start + length
        labels[:, end - 1] = IGNORE_INDEX
        position_ids[:, start:end] = torch.arange(length, device=input_ids.device, dtype=torch.long)
        sequence_ids[:, start:end] = row_sequence_bases + packed_idx
        start = end

    return {
        INPUT_IDS_KEY: input_ids,
        LABELS_KEY: labels,
        POSITION_IDS_KEY: position_ids.contiguous(),
        SEQUENCE_IDS_KEY: sequence_ids.contiguous(),
    }
