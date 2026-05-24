from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class PretrainingDataState:
    shard_idx: int
    token_offset: int
    consumed_tokens: int
    seed: int


class TokenShardDataset:
    """Memory-mapped token shards for causal LM pretraining."""

    def __init__(self, paths: Iterable[str | Path], *, dtype: np.dtype = np.uint32) -> None:
        self.paths = [Path(path) for path in paths]
        if not self.paths:
            raise ValueError("TokenShardDataset requires at least one token shard")
        self.dtype = np.dtype(dtype)
        self.shards = [np.memmap(path, mode="r", dtype=self.dtype) for path in self.paths]
        for path, shard in zip(self.paths, self.shards):
            if shard.size == 0:
                raise ValueError(f"token shard is empty: {path}")

    def __len__(self) -> int:
        return int(sum(shard.size for shard in self.shards))

    def read(self, shard_idx: int, token_offset: int, length: int) -> tuple[np.ndarray, int, int]:
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        tokens = np.empty(length, dtype=self.dtype)
        written = 0
        shard_idx %= len(self.shards)
        token_offset = int(token_offset)
        while written < length:
            shard = self.shards[shard_idx]
            if token_offset >= shard.size:
                shard_idx = (shard_idx + 1) % len(self.shards)
                token_offset = 0
                continue
            take = min(length - written, int(shard.size) - token_offset)
            tokens[written : written + take] = shard[token_offset : token_offset + take]
            written += take
            token_offset += take
            if token_offset >= shard.size and written < length:
                shard_idx = (shard_idx + 1) % len(self.shards)
                token_offset = 0
        return tokens, shard_idx, token_offset


class PretrainingDataLoader:
    """DP-aware deterministic token-stream loader for next-token prediction."""

    def __init__(
        self,
        dataset: TokenShardDataset,
        *,
        seq_len: int,
        micro_batch_size: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        seed: int = 1234,
        start_state: PretrainingDataState | None = None,
    ) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if micro_batch_size < 1:
            raise ValueError(f"micro_batch_size must be >= 1, got {micro_batch_size}")
        if dp_world_size < 1:
            raise ValueError(f"dp_world_size must be >= 1, got {dp_world_size}")
        if dp_rank < 0 or dp_rank >= dp_world_size:
            raise ValueError(f"dp_rank must be in [0, {dp_world_size}), got {dp_rank}")
        self.dataset = dataset
        self.seq_len = seq_len
        self.micro_batch_size = micro_batch_size
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.seed = seed
        self._tokens_per_sample = seq_len + 1
        self.shard_idx = 0
        self.token_offset = dp_rank * self._tokens_per_sample
        self.consumed_tokens = 0
        if start_state is not None:
            self.load_state_dict(asdict(start_state))

    def next_batch(self) -> dict[str, torch.Tensor]:
        samples = []
        for _ in range(self.micro_batch_size):
            sample, self.shard_idx, self.token_offset = self.dataset.read(
                self.shard_idx,
                self.token_offset,
                self._tokens_per_sample,
            )
            samples.append(sample)
            self._advance_to_next_dp_sample()
        tokens = torch.from_numpy(np.stack(samples).astype(np.int64, copy=False))
        self.consumed_tokens += self.micro_batch_size * self.seq_len
        return {
            "input_ids": tokens[:, :-1].contiguous(),
            "labels": tokens[:, 1:].contiguous(),
        }

    def state_dict(self) -> PretrainingDataState:
        return PretrainingDataState(
            shard_idx=self.shard_idx,
            token_offset=self.token_offset,
            consumed_tokens=self.consumed_tokens,
            seed=self.seed,
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        loader_state = PretrainingDataState(**state)
        if loader_state.shard_idx < 0 or loader_state.shard_idx >= len(self.dataset.shards):
            raise ValueError(f"invalid shard_idx={loader_state.shard_idx}")
        if loader_state.token_offset < 0 or loader_state.token_offset > self.dataset.shards[loader_state.shard_idx].size:
            raise ValueError(f"invalid token_offset={loader_state.token_offset}")
        self.shard_idx = loader_state.shard_idx
        self.token_offset = loader_state.token_offset
        self.consumed_tokens = loader_state.consumed_tokens
        self.seed = loader_state.seed

    def _advance_to_next_dp_sample(self) -> None:
        skip = (self.dp_world_size - 1) * self._tokens_per_sample
        if skip > 0:
            _, self.shard_idx, self.token_offset = self.dataset.read(
                self.shard_idx,
                self.token_offset,
                skip,
            )
