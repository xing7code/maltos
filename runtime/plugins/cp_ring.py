from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class RingAttentionCore(nn.Module):
    def __init__(self, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.group = group

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, position_offset: int) -> torch.Tensor:
        raise NotImplementedError("RingAttentionCore is not implemented yet")
