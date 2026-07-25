from __future__ import annotations

"""Canonical-to-local batch conversion for fixed context-parallel layouts.

The conversion deliberately lives in :mod:`parallel`, rather than in the CP
runtime plugin.  A CP-aware loader can therefore emit the exact same local
batch that the runtime would otherwise derive after host-to-device transfer.
"""

from dataclasses import dataclass
from typing import Any

import torch

from parallel.context_token_planner import ContextTokenPlanner
from utils.constants import (
    HIDDEN_STATES_KEY,
    IGNORE_INDEX,
    INPUT_IDS_KEY,
    LABELS_KEY,
    LOSS_WEIGHT_KEY,
    POSITION_IDS_KEY,
    POSITION_OFFSET_KEY,
    SEQUENCE_IDS_KEY,
)


@dataclass(frozen=True)
class ContextParallelBatchSharder:
    """Applies one fixed CP attention-core layout to a canonical batch."""

    planner: ContextTokenPlanner
    world_size: int

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError(f"CP world_size must be >= 1, got {self.world_size}")

    def shard(self, batch: Any, *, rank: int) -> Any:
        if rank < 0 or rank >= self.world_size:
            raise ValueError(f"CP rank must be in [0, {self.world_size}), got {rank}")
        if self.world_size == 1:
            return batch
        seq_len = _infer_seq_len(batch)
        plan = self.planner.plan(
            sequence_lengths=[seq_len] * _infer_batch_size(batch),
            world_size=self.world_size,
        )
        local_positions = plan.local_positions(rank)
        if isinstance(batch, dict):
            sharded = dict(batch)
            for key in (INPUT_IDS_KEY, LABELS_KEY, HIDDEN_STATES_KEY, POSITION_IDS_KEY, SEQUENCE_IDS_KEY):
                value = sharded.get(key)
                sharded[key] = _shard_batch_item(value, local_positions, seq_len)
            sharded[POSITION_IDS_KEY] = _materialize_position_ids(
                sharded.get(POSITION_IDS_KEY),
                positions=local_positions,
                seq_len=seq_len,
                reference=_batch_reference_tensor(sharded),
            )
            sharded[POSITION_OFFSET_KEY] = int(local_positions[0].item())
            sharded[LOSS_WEIGHT_KEY] = _loss_weight(batch.get(LABELS_KEY), sharded.get(LABELS_KEY))
            return sharded
        if isinstance(batch, (tuple, list)):
            input_ids = _shard_batch_item(batch[0], local_positions, seq_len) if len(batch) > 0 else None
            labels = _shard_batch_item(batch[1], local_positions, seq_len) if len(batch) > 1 else None
            return {
                INPUT_IDS_KEY: input_ids,
                LABELS_KEY: labels,
                POSITION_IDS_KEY: _materialize_position_ids(
                    None,
                    positions=local_positions,
                    seq_len=seq_len,
                    reference=input_ids if torch.is_tensor(input_ids) else labels,
                ),
                POSITION_OFFSET_KEY: int(local_positions[0].item()),
                LOSS_WEIGHT_KEY: _loss_weight(batch[1] if len(batch) > 1 else None, labels),
            }
        raise TypeError(f"ContextParallelBatchSharder does not support batch type={type(batch).__name__}")


def _infer_seq_len(batch: Any) -> int:
    if isinstance(batch, dict):
        for key in (INPUT_IDS_KEY, LABELS_KEY, HIDDEN_STATES_KEY, SEQUENCE_IDS_KEY):
            value = batch.get(key)
            if torch.is_tensor(value) and value.dim() >= 2:
                return int(value.size(1))
        position_ids = batch.get(POSITION_IDS_KEY)
        if torch.is_tensor(position_ids):
            if position_ids.dim() >= 2:
                return int(position_ids.size(1))
            if position_ids.dim() == 1:
                return int(position_ids.size(0))
        raise TypeError("ContextParallelBatchSharder could not infer sequence length from dict batch")
    if isinstance(batch, (tuple, list)):
        for value in batch:
            if torch.is_tensor(value) and value.dim() >= 2:
                return int(value.size(1))
        raise TypeError("ContextParallelBatchSharder could not infer sequence length from tuple/list batch")
    raise TypeError(f"ContextParallelBatchSharder does not support batch type={type(batch).__name__}")


def _infer_batch_size(batch: Any) -> int:
    values = batch.values() if isinstance(batch, dict) else batch if isinstance(batch, (tuple, list)) else ()
    for value in values:
        if torch.is_tensor(value) and value.dim() >= 2:
            return int(value.size(0))
    raise TypeError("ContextParallelBatchSharder could not infer batch size")


def _batch_reference_tensor(batch: dict[str, Any]) -> torch.Tensor | None:
    for key in (INPUT_IDS_KEY, LABELS_KEY, HIDDEN_STATES_KEY, SEQUENCE_IDS_KEY):
        value = batch.get(key)
        if torch.is_tensor(value) and value.dim() >= 2:
            return value
    return None


def _materialize_position_ids(
    current: Any,
    *,
    positions: torch.Tensor,
    seq_len: int,
    reference: torch.Tensor | None,
) -> torch.Tensor:
    device = reference.device if reference is not None else positions.device
    if torch.is_tensor(current):
        sharded = _shard_batch_item(current, positions, seq_len)
        if torch.is_tensor(sharded):
            return sharded.to(device=device, dtype=torch.long)
    if reference is None:
        return positions.to(dtype=torch.long)
    return positions.unsqueeze(0).expand(int(reference.size(0)), -1).contiguous().to(device=device, dtype=torch.long)


def _shard_batch_item(value: Any, positions: torch.Tensor, seq_len: int) -> Any:
    if torch.is_tensor(value):
        index = positions.to(device=value.device)
        if value.dim() >= 2 and value.size(1) == seq_len:
            return value.index_select(1, index).contiguous()
        if value.dim() == 1 and value.size(0) == seq_len:
            return value.index_select(0, index).contiguous()
    return value


def _loss_weight(full_labels: Any, local_labels: Any) -> float | None:
    if not torch.is_tensor(full_labels) or not torch.is_tensor(local_labels):
        return None
    full_count = int((full_labels != IGNORE_INDEX).sum().item())
    local_count = int((local_labels != IGNORE_INDEX).sum().item())
    if full_count == 0:
        return None
    return float(local_count) / float(full_count)
