from __future__ import annotations

import torch

from utils.constants import IGNORE_INDEX


def causal_lm_batch(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = IGNORE_INDEX
    return input_ids, labels
