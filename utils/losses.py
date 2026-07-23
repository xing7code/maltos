from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.constants import IGNORE_INDEX


def causal_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean causal CE that is finite when a packed microbatch has no targets."""
    flat_logits = logits.contiguous().view(-1, logits.size(-1))
    flat_labels = labels.contiguous().view(-1)
    loss_sum = F.cross_entropy(flat_logits, flat_labels, ignore_index=IGNORE_INDEX, reduction="sum")
    target_count = flat_labels.ne(IGNORE_INDEX).sum().clamp_min(1)
    return loss_sum / target_count
