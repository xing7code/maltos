from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SimpleDataLoaderState:
    cursor: int
    epoch: int
    consumed_tokens: int


class SimpleTensorDataLoader:
    """Tiny deterministic tensor batcher for tests with checkpointable cursor state."""

    def __init__(self, data: torch.Tensor, batch_size: int, *, drop_last: bool = True) -> None:
        if data.size(0) <= 0:
            raise ValueError("data must have a non-empty leading batch dimension")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if drop_last and batch_size > data.size(0):
            raise ValueError(f"batch_size={batch_size} exceeds dataset size={data.size(0)}")
        self.data = data
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.cursor = 0
        self.epoch = 0
        self.consumed_tokens = 0

    def next_batch(self) -> torch.Tensor:
        if self.cursor + self.batch_size > self.data.size(0):
            self.epoch += 1
            self.cursor = 0
        end = min(self.cursor + self.batch_size, self.data.size(0))
        if self.drop_last and end - self.cursor < self.batch_size:
            self.epoch += 1
            self.cursor = 0
            end = self.batch_size
        batch = self.data[self.cursor : end].contiguous()
        self.cursor = end
        self.consumed_tokens += batch.numel()
        return batch

    def state_dict(self) -> SimpleDataLoaderState:
        return SimpleDataLoaderState(
            cursor=self.cursor,
            epoch=self.epoch,
            consumed_tokens=self.consumed_tokens,
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        loader_state = SimpleDataLoaderState(**state)
        if loader_state.cursor < 0 or loader_state.cursor > self.data.size(0):
            raise ValueError(f"invalid dataloader cursor={loader_state.cursor}")
        self.cursor = loader_state.cursor
        self.epoch = loader_state.epoch
        self.consumed_tokens = loader_state.consumed_tokens
