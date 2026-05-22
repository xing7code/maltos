from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.distributed as dist
from train_system.state.registry import ParamState

if TYPE_CHECKING:
    from train_system.runtime.core import RuntimeCore


@dataclass(frozen=True)
class RankCheckpointMetadata:
    rank: int
    entries: list[ParamState]


@dataclass(frozen=True)
class CheckpointManifest:
    version: int
    world_size: int
    ranks: list[RankCheckpointMetadata]
    optimizer_source_ranks: list[int]
    artifacts: list["CheckpointArtifact"]


@dataclass(frozen=True)
class CheckpointArtifact:
    kind: Literal["model", "optimizer", "trainer"]
    rank: int
    path: str
    source_rank: int | None = None


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
    model_path = checkpoint_dir / f"model_rank_{rank}.pt"
    torch.save(model_state, model_path)
    local_artifacts: list[dict[str, Any]] = [
        asdict(CheckpointArtifact(kind="model", rank=rank, path=model_path.name)),
    ]
    if optim_state is not None and runtime.should_save_optimizer(rank):
        optim_path = checkpoint_dir / f"optim_rank_{rank}.pt"
        torch.save(optim_state, optim_path)
        local_artifacts.append(
            asdict(
                CheckpointArtifact(
                    kind="optimizer",
                    rank=rank,
                    path=optim_path.name,
                    source_rank=rank,
                )
            )
        )
    trainer_path = checkpoint_dir / f"trainer_rank_{rank}.pt"
    torch.save(_local_trainer_state(runtime).to_dict(), trainer_path)
    local_artifacts.append(asdict(CheckpointArtifact(kind="trainer", rank=rank, path=trainer_path.name)))

    gathered_metadata: list[list[dict] | None] = [None for _ in range(world_size)]
    gathered_artifacts: list[list[dict] | None] = [None for _ in range(world_size)]
    if dist.is_initialized():
        dist.all_gather_object(gathered_metadata, [asdict(item) for item in rank_entries])
        dist.all_gather_object(gathered_artifacts, local_artifacts)
    else:
        gathered_metadata[0] = [asdict(item) for item in rank_entries]
        gathered_artifacts[0] = local_artifacts

    if rank == 0:
        optimizer_source_ranks = [runtime.optimizer_state_source_rank(rank_id) for rank_id in range(world_size)]
        manifest = CheckpointManifest(
            version=1,
            world_size=world_size,
            ranks=[
                RankCheckpointMetadata(
                    rank=rank_idx,
                    entries=[ParamState(**item) for item in (entries or [])],
                )
                for rank_idx, entries in enumerate(gathered_metadata)
            ],
            optimizer_source_ranks=optimizer_source_ranks,
            artifacts=[
                CheckpointArtifact(**item)
                for rank_artifacts in gathered_artifacts
                for item in (rank_artifacts or [])
            ],
        )
        with open(checkpoint_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2)

    if dist.is_initialized():
        dist.barrier()


def load_sharded_checkpoint(runtime: "RuntimeCore", path: str | Path) -> None:
    checkpoint_dir = Path(path)
    rank = dist.get_rank() if dist.is_initialized() else 0
    manifest = _load_manifest(checkpoint_dir)
    _validate_manifest_for_runtime(manifest, runtime)

    model_artifact = _find_artifact(manifest, kind="model", rank=rank)
    if model_artifact is None:
        raise ValueError(f"manifest missing model artifact for rank={rank}")
    model_state = torch.load(checkpoint_dir / model_artifact.path, map_location="cpu")

    trainer_artifact = _find_artifact(manifest, kind="trainer", rank=rank)
    if trainer_artifact is None:
        raise ValueError(f"manifest missing trainer artifact for rank={rank}")
    trainer_state = torch.load(checkpoint_dir / trainer_artifact.path, map_location="cpu")

    optim_source_rank = manifest.optimizer_source_ranks[rank]
    optimizer_artifact = _find_artifact(manifest, kind="optimizer", rank=optim_source_rank)
    optim_state = (
        torch.load(checkpoint_dir / optimizer_artifact.path, map_location="cpu")
        if optimizer_artifact is not None
        else None
    )
    _load_local_checkpoint_state(runtime, model_state, optim_state, trainer_state)
    if dist.is_initialized():
        dist.barrier()


def _load_manifest(checkpoint_dir: Path) -> CheckpointManifest:
    manifest_path = checkpoint_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    version = int(raw.get("version", 0))
    if version != 1:
        raise ValueError(f"unsupported checkpoint manifest version={version}")
    if "artifacts" not in raw:
        raise ValueError("manifest missing artifacts field")
    artifact_items = raw["artifacts"]
    return CheckpointManifest(
        version=version,
        world_size=int(raw["world_size"]),
        ranks=[
            RankCheckpointMetadata(
                rank=int(rank_meta["rank"]),
                entries=[ParamState(**entry) for entry in rank_meta["entries"]],
            )
            for rank_meta in raw["ranks"]
        ],
        optimizer_source_ranks=[int(rank_id) for rank_id in raw["optimizer_source_ranks"]],
        artifacts=[
            CheckpointArtifact(
                kind=item["kind"],
                rank=int(item["rank"]),
                path=item["path"],
                source_rank=int(item["source_rank"]) if item.get("source_rank") is not None else None,
            )
            for item in artifact_items
        ],
    )


def _find_artifact(
    manifest: CheckpointManifest,
    kind: Literal["model", "optimizer", "trainer"],
    rank: int,
) -> CheckpointArtifact | None:
    for artifact in manifest.artifacts:
        if artifact.kind == kind and artifact.rank == rank:
            return artifact
    return None


def _validate_manifest_for_runtime(manifest: CheckpointManifest, runtime: "RuntimeCore") -> None:
    runtime_world_size = dist.get_world_size() if dist.is_initialized() else 1
    runtime_rank = dist.get_rank() if dist.is_initialized() else 0
    if manifest.world_size != runtime_world_size:
        raise ValueError(
            f"checkpoint world_size mismatch: checkpoint={manifest.world_size}, runtime={runtime_world_size}"
        )
    if len(manifest.ranks) != manifest.world_size:
        raise ValueError(
            f"manifest ranks length mismatch: expected={manifest.world_size}, got={len(manifest.ranks)}"
        )
    if len(manifest.optimizer_source_ranks) != manifest.world_size:
        raise ValueError(
            "manifest optimizer_source_ranks length mismatch: "
            f"expected={manifest.world_size}, got={len(manifest.optimizer_source_ranks)}"
        )
    if runtime_rank >= manifest.world_size:
        raise ValueError(f"runtime rank out of range: rank={runtime_rank}, world_size={manifest.world_size}")

    for rank_id in range(manifest.world_size):
        if _find_artifact(manifest, kind="model", rank=rank_id) is None:
            raise ValueError(f"manifest missing model artifact for rank={rank_id}")
        if _find_artifact(manifest, kind="trainer", rank=rank_id) is None:
            raise ValueError(f"manifest missing trainer artifact for rank={rank_id}")
    _validate_unique_artifacts(manifest)

    optimizer_artifact_ranks = {artifact.rank for artifact in manifest.artifacts if artifact.kind == "optimizer"}
    if optimizer_artifact_ranks:
        for rank_id, source_rank in enumerate(manifest.optimizer_source_ranks):
            if source_rank < 0 or source_rank >= manifest.world_size:
                raise ValueError(
                    f"manifest optimizer source rank out of range: rank={rank_id}, source_rank={source_rank}"
                )
            if source_rank not in optimizer_artifact_ranks:
                raise ValueError(
                    f"manifest missing optimizer artifact for source rank={source_rank} (used by rank={rank_id})"
                )
        for rank_id in range(manifest.world_size):
            runtime_source_rank = runtime.optimizer_state_source_rank(rank_id)
            if runtime_source_rank != manifest.optimizer_source_ranks[rank_id]:
                raise ValueError(
                    "optimizer source mapping mismatch between runtime and checkpoint manifest: "
                    f"rank={rank_id}, runtime={runtime_source_rank}, "
                    f"checkpoint={manifest.optimizer_source_ranks[rank_id]}"
                )


def _validate_unique_artifacts(manifest: CheckpointManifest) -> None:
    seen: set[tuple[str, int]] = set()
    for artifact in manifest.artifacts:
        key = (artifact.kind, artifact.rank)
        if key in seen:
            raise ValueError(f"duplicate artifact in manifest: kind={artifact.kind}, rank={artifact.rank}")
        seen.add(key)


_OPTIMIZER_STATE_PREFIX = "__optimizer_state__."
_RUNTIME_OPTIMIZER_STATE_KEY = f"{_OPTIMIZER_STATE_PREFIX}runtime"
_SCHEDULER_STATE_PREFIX = "__scheduler_state__."
_RUNTIME_SCHEDULER_STATE_KEY = f"{_SCHEDULER_STATE_PREFIX}runtime"


def _local_checkpoint_state(
    runtime: "RuntimeCore",
) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None, list[ParamState]]:
    for plugin in runtime.plugins:
        local_state_dict = getattr(plugin, "local_state_dict", None)
        if local_state_dict is not None:
            state, entries = local_state_dict()
            break
    else:
        state = {}
        entries = []
        for name, param_state in runtime.state_registry.iter_param_states():
            param = runtime.state_registry.get_param_tensor(name)
            tensor = param.detach().cpu().clone()
            state[name] = tensor
            entries.append(param_state)

    for entry in entries:
        for plugin in runtime.plugins:
            annotate = getattr(plugin, "annotate_checkpoint_state", None)
            if annotate is not None:
                annotate(entry)
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
        for name, tensor in model_state.items():
            param_state = runtime.state_registry.get_param_state(name)
            param = runtime.state_registry.get_param_tensor(name)
            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))

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
