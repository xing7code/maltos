from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class SFTDataState:
    shard_idx: int
    row_offset: int
    consumed_tokens: int
    consumed_sequences: int
    seed: int


@dataclass(frozen=True)
class _SFTFieldSpec:
    dtype: np.dtype
    offset: int
    shape: tuple[int, int]


@dataclass(frozen=True)
class _SFTShardArrays:
    path: Path
    sequences: int
    input_ids: np.memmap
    labels: np.memmap
    position_ids: np.memmap
    sequence_ids: np.memmap


class PackedSFTDataset:
    """Memory-mapped packed SFT shards with per-field offsets from meta.json."""

    def __init__(self, root_or_meta: str | Path) -> None:
        self.meta_path = _resolve_meta_path(root_or_meta)
        self.root = self.meta_path.parent
        payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if payload.get("format") != "maltos_sft_packed":
            raise ValueError(f"unsupported SFT dataset format={payload.get('format')!r}")

        packing = payload.get("packing")
        if isinstance(packing, dict):
            layout = packing.get("layout") or {}
            shards = packing.get("shards") or []
        else:
            layout = payload
            shards = payload.get("shards") or []
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"SFT dataset meta must define non-empty shards, got {type(shards)!r}")

        self.seq_len = int(layout.get("seq_len", payload.get("seq_len", 0)))
        if self.seq_len < 1:
            raise ValueError(f"SFT dataset seq_len must be >= 1, got {self.seq_len}")
        self.ignore_index = int(_nested_get(layout, ("fields", "labels", "ignore_index"), default=-100))
        self.pad_sequence_id = int(_nested_get(layout, ("fields", "sequence_ids", "pad_sequence_id"), default=-1))

        self.shards: list[_SFTShardArrays] = []
        for shard_payload in shards:
            shard = self._load_shard(shard_payload)
            self.shards.append(shard)
        self.shard_paths = [shard.path for shard in self.shards]
        self.total_sequences = int(sum(shard.sequences for shard in self.shards))
        if self.total_sequences < 1:
            raise ValueError("SFT dataset must contain at least one packed sequence")

    def __len__(self) -> int:
        return self.total_sequences

    def advance(self, shard_idx: int, row_offset: int, rows: int) -> tuple[int, int]:
        if rows < 0:
            raise ValueError(f"rows must be >= 0, got {rows}")
        shard_idx %= len(self.shards)
        row_offset = int(row_offset)
        remaining = int(rows)
        while remaining > 0:
            shard = self.shards[shard_idx]
            if row_offset >= shard.sequences:
                shard_idx = (shard_idx + 1) % len(self.shards)
                row_offset = 0
                continue
            available = shard.sequences - row_offset
            if remaining < available:
                row_offset += remaining
                remaining = 0
                break
            remaining -= available
            shard_idx = (shard_idx + 1) % len(self.shards)
            row_offset = 0
        return shard_idx, row_offset

    def read_rows(
        self,
        shard_idx: int,
        row_offset: int,
        rows: int,
    ) -> tuple[dict[str, np.ndarray], int, int]:
        if rows < 1:
            raise ValueError(f"rows must be >= 1, got {rows}")
        inputs = np.empty((rows, self.seq_len), dtype=self.shards[0].input_ids.dtype)
        labels = np.empty((rows, self.seq_len), dtype=self.shards[0].labels.dtype)
        positions = np.empty((rows, self.seq_len), dtype=self.shards[0].position_ids.dtype)
        sequences = np.empty((rows, self.seq_len), dtype=self.shards[0].sequence_ids.dtype)

        written = 0
        shard_idx %= len(self.shards)
        row_offset = int(row_offset)
        while written < rows:
            shard = self.shards[shard_idx]
            if row_offset >= shard.sequences:
                shard_idx = (shard_idx + 1) % len(self.shards)
                row_offset = 0
                continue
            take = min(rows - written, shard.sequences - row_offset)
            src = slice(row_offset, row_offset + take)
            dst = slice(written, written + take)
            inputs[dst] = shard.input_ids[src]
            labels[dst] = shard.labels[src]
            positions[dst] = shard.position_ids[src]
            sequences[dst] = shard.sequence_ids[src]
            written += take
            row_offset += take
            if row_offset >= shard.sequences and written < rows:
                shard_idx = (shard_idx + 1) % len(self.shards)
                row_offset = 0
        return (
            {
                "input_ids": inputs,
                "labels": labels,
                "position_ids": positions,
                "sequence_ids": sequences,
            },
            shard_idx,
            row_offset,
        )

    def _load_shard(self, shard_payload: dict[str, Any]) -> _SFTShardArrays:
        path = self.root / str(shard_payload["path"])
        fields = shard_payload.get("fields")
        if not isinstance(fields, dict):
            raise ValueError(f"invalid shard fields for {path}: {type(fields)!r}")
        input_spec = _parse_field_spec(fields.get("input_ids"))
        label_spec = _parse_field_spec(fields.get("labels"))
        position_spec = _parse_field_spec(fields.get("position_ids"))
        sequence_spec = _parse_field_spec(fields.get("sequence_ids"))
        sequences = int(shard_payload["sequences"])
        return _SFTShardArrays(
            path=path,
            sequences=sequences,
            input_ids=np.memmap(path, mode="r", dtype=input_spec.dtype, offset=input_spec.offset, shape=input_spec.shape),
            labels=np.memmap(path, mode="r", dtype=label_spec.dtype, offset=label_spec.offset, shape=label_spec.shape),
            position_ids=np.memmap(path, mode="r", dtype=position_spec.dtype, offset=position_spec.offset, shape=position_spec.shape),
            sequence_ids=np.memmap(path, mode="r", dtype=sequence_spec.dtype, offset=sequence_spec.offset, shape=sequence_spec.shape),
        )


class SFTDataLoader:
    """DP-aware deterministic row loader for packed SFT sequences."""

    def __init__(
        self,
        dataset: PackedSFTDataset,
        *,
        micro_batch_size: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        seed: int = 1234,
        start_state: SFTDataState | None = None,
    ) -> None:
        if micro_batch_size < 1:
            raise ValueError(f"micro_batch_size must be >= 1, got {micro_batch_size}")
        if dp_world_size < 1:
            raise ValueError(f"dp_world_size must be >= 1, got {dp_world_size}")
        if dp_rank < 0 or dp_rank >= dp_world_size:
            raise ValueError(f"dp_rank must be in [0, {dp_world_size}), got {dp_rank}")
        self.dataset = dataset
        self.micro_batch_size = micro_batch_size
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.seed = seed
        self.shard_idx, self.row_offset = self.dataset.advance(0, 0, dp_rank)
        self.consumed_sequences = 0
        self.consumed_tokens = 0
        if start_state is not None:
            self.load_state_dict(asdict(start_state))

    def next_batch(self) -> dict[str, torch.Tensor]:
        rows: dict[str, list[np.ndarray]] = {
            "input_ids": [],
            "labels": [],
            "position_ids": [],
            "sequence_ids": [],
        }
        for _ in range(self.micro_batch_size):
            sample, self.shard_idx, self.row_offset = self.dataset.read_rows(
                self.shard_idx,
                self.row_offset,
                1,
            )
            for key in rows:
                rows[key].append(sample[key][0])
            self.shard_idx, self.row_offset = self.dataset.advance(
                self.shard_idx,
                self.row_offset,
                self.dp_world_size - 1,
            )

        batch = {
            key: torch.from_numpy(np.stack(values).astype(np.int64, copy=False)).contiguous()
            for key, values in rows.items()
        }
        self.consumed_sequences += self.micro_batch_size
        self.consumed_tokens += self.micro_batch_size * self.dataset.seq_len
        return batch

    def state_dict(self) -> SFTDataState:
        return SFTDataState(
            shard_idx=self.shard_idx,
            row_offset=self.row_offset,
            consumed_tokens=self.consumed_tokens,
            consumed_sequences=self.consumed_sequences,
            seed=self.seed,
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        loader_state = SFTDataState(**state)
        if loader_state.shard_idx < 0 or loader_state.shard_idx >= len(self.dataset.shards):
            raise ValueError(f"invalid shard_idx={loader_state.shard_idx}")
        shard = self.dataset.shards[loader_state.shard_idx]
        if loader_state.row_offset < 0 or loader_state.row_offset > shard.sequences:
            raise ValueError(f"invalid row_offset={loader_state.row_offset}")
        self.shard_idx = loader_state.shard_idx
        self.row_offset = loader_state.row_offset
        self.consumed_tokens = loader_state.consumed_tokens
        self.consumed_sequences = loader_state.consumed_sequences
        self.seed = loader_state.seed


def _resolve_meta_path(root_or_meta: str | Path) -> Path:
    path = Path(root_or_meta)
    if path.is_dir():
        path = path / "meta.json"
    if not path.is_file():
        raise ValueError(f"SFT dataset meta.json not found at {path}")
    return path


def _parse_field_spec(payload: Any) -> _SFTFieldSpec:
    if not isinstance(payload, dict):
        raise ValueError(f"invalid SFT field spec: {payload!r}")
    dtype = np.dtype(str(payload["dtype"]))
    offset = int(payload["offset"])
    shape_raw = payload["shape"]
    if not isinstance(shape_raw, list) or len(shape_raw) != 2:
        raise ValueError(f"invalid SFT field shape: {shape_raw!r}")
    shape = (int(shape_raw[0]), int(shape_raw[1]))
    return _SFTFieldSpec(dtype=dtype, offset=offset, shape=shape)


def _nested_get(payload: dict[str, Any], keys: tuple[str, ...], *, default: Any) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
