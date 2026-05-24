from __future__ import annotations

import torch
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        out = self.net(batch)
        return out.pow(2).mean()

