from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from train_system.runtime.core import RuntimeCore
    from train_system.runtime.plugin import RuntimePlugin


@dataclass
class CheckpointEntry:
    state_key: str
    logical_names: list[str]
    logical_shapes: list[tuple[int, ...]]
    physical_shape: tuple[int, ...]
    dtype: str
    annotations: dict[str, object] = field(default_factory=dict)

    def add_annotation(self, plugin: "RuntimePlugin") -> None:
        annotate = getattr(plugin, "annotate_checkpoint_entry", None)
        if annotate is not None:
            annotate(self)

    def set_plugin_annotation(self, plugin: "RuntimePlugin", value: object) -> None:
        key = plugin.id.value
        if key in self.annotations:
            raise ValueError(f"duplicate checkpoint annotation: {key}")
        self.annotations[key] = value


@dataclass(frozen=True)
class RankCheckpointMetadata:
    rank: int
    entries: list[CheckpointEntry]


@dataclass(frozen=True)
class CheckpointManifest:
    world_size: int
    ranks: list[RankCheckpointMetadata]


def save_sharded_checkpoint(runtime: "RuntimeCore", path: str | Path) -> None:
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    local_state, rank_entries = _local_checkpoint_state(runtime)
    torch.save(local_state, checkpoint_dir / f"rank_{rank}.pt")

    gathered_metadata: list[list[dict] | None] = [None for _ in range(world_size)]
    if dist.is_initialized():
        dist.all_gather_object(gathered_metadata, [asdict(item) for item in rank_entries])
    else:
        gathered_metadata[0] = [asdict(item) for item in rank_entries]

    if rank == 0:
        manifest = CheckpointManifest(
            world_size=world_size,
            ranks=[
                RankCheckpointMetadata(
                    rank=rank_idx,
                    entries=[CheckpointEntry(**item) for item in (entries or [])],
                )
                for rank_idx, entries in enumerate(gathered_metadata)
            ],
        )
        with open(checkpoint_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2)

    if dist.is_initialized():
        dist.barrier()


def load_sharded_checkpoint(runtime: "RuntimeCore", path: str | Path) -> None:
    checkpoint_dir = Path(path)
    rank = dist.get_rank() if dist.is_initialized() else 0
    state = torch.load(checkpoint_dir / f"rank_{rank}.pt", map_location="cpu")
    _load_local_checkpoint_state(runtime, state)
    if dist.is_initialized():
        dist.barrier()


def _local_checkpoint_state(runtime: "RuntimeCore") -> tuple[dict[str, torch.Tensor], list[CheckpointEntry]]:
    for plugin in runtime.plugins:
        local_state_dict = getattr(plugin, "local_state_dict", None)
        if local_state_dict is not None:
            state, entries = local_state_dict()
            break
    else:
        state = {}
        entries = []
        for name, param in runtime.model.named_parameters():
            tensor = param.detach().cpu().clone()
            state[name] = tensor
            entry = CheckpointEntry(
                state_key=name,
                logical_names=[name],
                logical_shapes=[tuple(param.shape)],
                physical_shape=tuple(param.shape),
                dtype=str(param.dtype),
                annotations={},
            )
            entries.append(entry)

    for entry in entries:
        for plugin in runtime.plugins:
            entry.add_annotation(plugin)
    return state, entries


def _load_local_checkpoint_state(runtime: "RuntimeCore", state: dict[str, torch.Tensor]) -> None:
    for plugin in runtime.plugins:
        load_local_state_dict = getattr(plugin, "load_local_state_dict", None)
        if load_local_state_dict is not None:
            load_local_state_dict(state)
            return

    params = dict(runtime.model.named_parameters())
    for name, tensor in state.items():
        params[name].data.copy_(tensor.to(device=params[name].device, dtype=params[name].dtype))
