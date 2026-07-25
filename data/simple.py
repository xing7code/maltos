from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from parallel.context_batch import ContextParallelBatchSharder

@dataclass(frozen=True)
class SimpleDataLoaderState:
    cursor: int
    epoch: int
    consumed_tokens: int


class SimpleTensorDataLoader:
    """Tiny deterministic tensor batcher for tests with checkpointable cursor state."""

    def __init__(
        self,
        data: torch.Tensor,
        batch_size: int,
        *,
        drop_last: bool = True,
        cp_batch_sharder: ContextParallelBatchSharder | None = None,
    ) -> None:
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
        self._cp_batch_sharder = cp_batch_sharder

    def next_batch(self, cp_rank: int | None = None) -> torch.Tensor | dict[str, torch.Tensor]:
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
        if cp_rank is None:
            return batch
        if self._cp_batch_sharder is None:
            raise RuntimeError("CP-local next_batch(cp_rank) requires constructor cp_batch_sharder")
        return self._cp_batch_sharder.shard((batch,), rank=cp_rank)

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
