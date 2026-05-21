from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


@dataclass(frozen=True)
class RngCheckpointState:
    cpu: torch.Tensor
    cuda: list[torch.Tensor] | None = None

    def to_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {"cpu": self.cpu}
        if self.cuda is not None:
            state["cuda"] = self.cuda
        return state

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "RngCheckpointState":
        return cls(cpu=state["cpu"], cuda=state.get("cuda"))


@dataclass(frozen=True)
class TrainerCheckpointState:
    step: int
    rng: RngCheckpointState
    consumed_tokens: int | None = None
    dataloader: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "consumed_tokens": self.consumed_tokens,
            "dataloader": self.dataloader,
            "rng": self.rng.to_dict(),
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "TrainerCheckpointState":
        return cls(
            step=int(state.get("step", 0)),
            consumed_tokens=state.get("consumed_tokens"),
            dataloader=state.get("dataloader"),
            rng=RngCheckpointState.from_dict(state["rng"]),
        )


def save_sharded_checkpoint(runtime: "RuntimeCore", path: str | Path) -> None:
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    model_state, optim_state, rank_entries = _local_checkpoint_state(runtime)
    torch.save(model_state, checkpoint_dir / f"model_rank_{rank}.pt")
    if optim_state is not None and runtime.should_save_optimizer(rank):
        torch.save(optim_state, checkpoint_dir / f"optim_rank_{rank}.pt")
    torch.save(_local_trainer_state(runtime).to_dict(), checkpoint_dir / f"trainer_rank_{rank}.pt")

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
    model_state = torch.load(checkpoint_dir / f"model_rank_{rank}.pt", map_location="cpu")
    optim_rank = runtime.optimizer_state_source_rank(rank)
    optim_path = checkpoint_dir / f"optim_rank_{optim_rank}.pt"
    optim_state = torch.load(optim_path, map_location="cpu") if optim_path.exists() else None
    trainer_path = checkpoint_dir / f"trainer_rank_{rank}.pt"
    trainer_state = torch.load(trainer_path, map_location="cpu") if trainer_path.exists() else None
    _load_local_checkpoint_state(runtime, model_state, optim_state, trainer_state)
    if dist.is_initialized():
        dist.barrier()


_OPTIMIZER_STATE_PREFIX = "__optimizer_state__."
_RUNTIME_OPTIMIZER_STATE_KEY = f"{_OPTIMIZER_STATE_PREFIX}runtime"
_SCHEDULER_STATE_PREFIX = "__scheduler_state__."
_RUNTIME_SCHEDULER_STATE_KEY = f"{_SCHEDULER_STATE_PREFIX}runtime"


def _local_checkpoint_state(runtime: "RuntimeCore") -> tuple[dict[str, torch.Tensor], dict[str, Any] | None, list[CheckpointEntry]]:
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
    optim_state = _local_optimizer_state(runtime)
    return state, optim_state, entries


def _load_local_checkpoint_state(
    runtime: "RuntimeCore",
    model_state: dict[str, torch.Tensor],
    optim_state: dict[str, Any] | None,
    trainer_state: dict[str, Any] | None,
) -> None:
    for plugin in runtime.plugins:
        load_local_state_dict = getattr(plugin, "load_local_state_dict", None)
        if load_local_state_dict is not None:
            load_local_state_dict(model_state)
            break
    else:
        params = dict(runtime.model.named_parameters())
        for name, tensor in model_state.items():
            params[name].data.copy_(tensor.to(device=params[name].device, dtype=params[name].dtype))

    from train_system.runtime.core import RuntimePhase

    if optim_state is not None:
        _load_optimizer_states(runtime, optim_state)
    if trainer_state is not None:
        _load_trainer_state(runtime, TrainerCheckpointState.from_dict(trainer_state))
    runtime._run_phase(RuntimePhase.POST_LOAD)


def _local_optimizer_state(runtime: "RuntimeCore") -> dict[str, Any] | None:
    if runtime.optimizer is not None:
        state: dict[str, Any] = {_RUNTIME_OPTIMIZER_STATE_KEY: runtime.optimizer.state_dict()}
        if runtime.scheduler is not None:
            state[_RUNTIME_SCHEDULER_STATE_KEY] = runtime.scheduler.state_dict()
        return state
    for plugin in runtime.plugins:
        if not plugin.owns_optimizer:
            continue
        optimizer = getattr(plugin, "optimizer", None)
        if optimizer is not None:
            state = {f"{_OPTIMIZER_STATE_PREFIX}{plugin.id.value}": optimizer.state_dict()}
            scheduler = getattr(plugin, "scheduler", None)
            if scheduler is not None:
                state[f"{_SCHEDULER_STATE_PREFIX}{plugin.id.value}"] = scheduler.state_dict()
            return state
    return None


def _load_optimizer_states(runtime: "RuntimeCore", state: dict[str, Any]) -> None:
    if runtime.optimizer is not None:
        optimizer_state = state.get(_RUNTIME_OPTIMIZER_STATE_KEY)
        if optimizer_state is not None:
            runtime.optimizer.load_state_dict(optimizer_state)
        scheduler_state = state.get(_RUNTIME_SCHEDULER_STATE_KEY)
        if scheduler_state is not None and runtime.scheduler is not None:
            runtime.scheduler.load_state_dict(scheduler_state)
        return
    for plugin in runtime.plugins:
        if not plugin.owns_optimizer:
            continue
        optimizer = getattr(plugin, "optimizer", None)
        if optimizer is None:
            continue
        optimizer_state = state.get(f"{_OPTIMIZER_STATE_PREFIX}{plugin.id.value}")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            scheduler = getattr(plugin, "scheduler", None)
            scheduler_state = state.get(f"{_SCHEDULER_STATE_PREFIX}{plugin.id.value}")
            if scheduler is not None and scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            return
    raise ValueError("no optimizer state found")


def _local_trainer_state(runtime: "RuntimeCore") -> TrainerCheckpointState:
    rng_state = RngCheckpointState(
        cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    return TrainerCheckpointState(
        step=runtime.state.step,
        consumed_tokens=runtime.state.metadata.get("consumed_tokens"),
        dataloader=runtime.state.metadata.get("dataloader"),
        rng=rng_state,
    )


def _load_trainer_state(runtime: "RuntimeCore", state: TrainerCheckpointState) -> None:
    runtime.state.step = state.step
    runtime.state.metadata["consumed_tokens"] = state.consumed_tokens
    runtime.state.metadata["dataloader"] = state.dataloader

    torch.set_rng_state(state.rng.cpu)
    if state.rng.cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.rng.cuda)
