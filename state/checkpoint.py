# Checkpoint ownership model for param and optimizer saving.
#
# Legend: each cell is "param / optim", where
#   R = replicate
#   S = shard
#
# Shared params (shared trunk, current ZeRO semantics use DCP = DP x CP):
#
#   +-------+--------+--------+--------+--------+
#   | Axis  |  DDP   | ZeRO1  | ZeRO2  | ZeRO3  |
#   +-------+--------+--------+--------+--------+
#   | DCP   | R / R  | R / S  | R / S  | S / S  |
#   | TP/PP | S / S  | S / S  | S / S  | S / S  |
#   +-------+--------+--------+--------+--------+
#
# Expert params (assuming no TP-on-expert):
#
#   +------+--------+--------+--------+--------+
#   | Axis |  DDP   | ZeRO1  | ZeRO2  | ZeRO3  |
#   +------+--------+--------+--------+--------+
#   | EP   | S / S  | S / S  | S / S  | S / S  |
#   | EREP | R / R  | R / S  | R / S  | S / S  |
#   | PP   | S / S  | S / S  | S / S  | S / S  |
#   +------+--------+--------+--------+--------+
#
# ModelStateMeta.source_rank records which global rank owns the checkpoint payload
# for each logical entry. Replicated entries collapse to rank 0 of their
# replica group; sharded entries use their local rank. Optimizer ownership is
# still resolved separately because optimizer state is currently treated as one
# owned object, not per-param.
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
from state.state import (
    ModelStateMeta,
    OptimizerState,
    RngState,
    StateManager,
    TrainerState,
)
from runtime.types import PpStatus, RuntimePhase
from utils.distributed import distributed_barrier


@dataclass(frozen=True)
class RankCheckpointMetadata:
    rank: int
    entries: list[ModelStateMeta]


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


def _model_state_meta_from_dict(item: dict[str, Any]) -> ModelStateMeta:
    return ModelStateMeta(
        state_key=item["state_key"],
        logical_names=[str(name) for name in item["logical_names"]],
        logical_shapes=[tuple(shape) for shape in item["logical_shapes"]],
        physical_shape=tuple(item["physical_shape"]),
        dtype=item["dtype"],
        source_rank=int(item["source_rank"]) if item.get("source_rank") is not None else None,
    )


def save_sharded_checkpoint(
    state_manager: StateManager,
    path: str | Path,
    *,
    min_free_gb: float | None = None,
) -> None:
    _save_sharded_checkpoint_impl(
        state_manager,
        path,
        min_free_gb=min_free_gb,
        model_state_override=None,
        rank_entries_override=None,
        include_training_state=True,
    )


def save_sharded_checkpoint_from_model_state(
    state_manager: StateManager,
    path: str | Path,
    *,
    model_state: dict[str, torch.Tensor],
    rank_entries: list[ModelStateMeta],
    include_training_state: bool = True,
    min_free_gb: float | None = None,
) -> None:
    _save_sharded_checkpoint_impl(
        state_manager,
        path,
        min_free_gb=min_free_gb,
        model_state_override=model_state,
        rank_entries_override=rank_entries,
        include_training_state=include_training_state,
    )


def _save_sharded_checkpoint_impl(
    state_manager: StateManager,
    path: str | Path,
    *,
    min_free_gb: float | None,
    model_state_override: dict[str, torch.Tensor] | None,
    rank_entries_override: list[ModelStateMeta] | None,
    include_training_state: bool,
) -> None:
    checkpoint_dir = Path(path)
    if min_free_gb is not None:
        _check_min_free_space(checkpoint_dir.parent, min_free_gb)
    tmp_dir = checkpoint_dir.with_name(f"{checkpoint_dir.name}.tmp")
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise FileExistsError(f"checkpoint already exists: {checkpoint_dir}")
        if checkpoint_dir.exists():
            checkpoint_dir.rmdir()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.parent.mkdir(parents=True, exist_ok=True)
    distributed_barrier()
    _save_sharded_checkpoint_contents(
        state_manager,
        tmp_dir,
        model_state_override=model_state_override,
        rank_entries_override=rank_entries_override,
        include_training_state=include_training_state,
    )
    distributed_barrier()
    if rank == 0:
        tmp_dir.rename(checkpoint_dir)
    distributed_barrier()


def _save_sharded_checkpoint_contents(
    state_manager: StateManager,
    checkpoint_dir: Path,
    *,
    model_state_override: dict[str, torch.Tensor] | None = None,
    rank_entries_override: list[ModelStateMeta] | None = None,
    include_training_state: bool = True,
) -> None:
    runtime = state_manager.runtime
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if model_state_override is None or rank_entries_override is None:
        model_state, rank_entries = state_manager.export_model_state()
    else:
        model_state = {
            key: tensor.detach().cpu().contiguous()
            for key, tensor in model_state_override.items()
        }
        rank_entries = [
            ModelStateMeta(
                state_key=entry.state_key,
                logical_names=list(entry.logical_names),
                logical_shapes=list(entry.logical_shapes),
                physical_shape=tuple(entry.physical_shape),
                dtype=entry.dtype,
                source_rank=entry.source_rank,
            )
            for entry in rank_entries_override
        ]
    optimizer_state = state_manager.export_optimizer_state() if include_training_state else None
    optim_state = optimizer_state.state if optimizer_state is not None else None
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
    if include_training_state:
        trainer_path = checkpoint_dir / f"trainer_rank_{rank}.pt"
        torch.save(asdict(state_manager.export_trainer_state()), trainer_path)
        local_artifacts.append(asdict(CheckpointArtifact(kind="trainer", rank=rank, path=trainer_path.name)))

    gathered_metadata: list[list[dict] | None] | None = [None for _ in range(world_size)] if rank == 0 else None
    gathered_artifacts: list[list[dict] | None] | None = [None for _ in range(world_size)] if rank == 0 else None
    if dist.is_initialized():
        # TODO: support async checkpointing by snapshotting state first, then
        # decoupling manifest assembly / file I/O from the training critical path.
        dist.gather_object([asdict(item) for item in rank_entries], gathered_metadata, dst=0)
        dist.gather_object(local_artifacts, gathered_artifacts, dst=0)
    else:
        assert gathered_metadata is not None
        assert gathered_artifacts is not None
        gathered_metadata[0] = [asdict(item) for item in rank_entries]
        gathered_artifacts[0] = local_artifacts

    if rank == 0:
        assert gathered_metadata is not None
        assert gathered_artifacts is not None
        optimizer_source_ranks = [runtime.optimizer_checkpoint_rank(rank_id) for rank_id in range(world_size)]
        manifest = CheckpointManifest(
            version=1,
            world_size=world_size,
            ranks=[
                RankCheckpointMetadata(
                    rank=rank_idx,
                    entries=[_model_state_meta_from_dict(item) for item in (entries or [])],
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

    distributed_barrier()

def load_sharded_checkpoint(
    state_manager: StateManager,
    path: str | Path,
    *,
    weights_only: bool = False,
) -> None:
    runtime = state_manager.runtime

    checkpoint_dir = Path(path)
    rank = dist.get_rank() if dist.is_initialized() else 0
    manifest = load_checkpoint_manifest(checkpoint_dir)
    _validate_manifest_for_runtime(manifest, runtime, weights_only=weights_only)

    model_state = _load_model_state_for_rank(
        checkpoint_dir,
        manifest,
        rank_metadata=manifest.ranks[rank],
        rank=rank,
    )

    trainer_state = None
    optim_state = None
    if not weights_only:
        trainer_artifact = _find_artifact(manifest, kind="trainer", rank=rank)
        if trainer_artifact is None:
            raise ValueError(f"manifest missing trainer artifact for rank={rank}")
        with torch.serialization.safe_globals([PpStatus]):
            trainer_state = torch.load(checkpoint_dir / trainer_artifact.path, map_location="cpu", weights_only=True)

        optim_source_rank = manifest.optimizer_source_ranks[rank]
        optimizer_artifact = _find_artifact(manifest, kind="optimizer", rank=optim_source_rank)
        optim_state = (
            torch.load(checkpoint_dir / optimizer_artifact.path, map_location="cpu", weights_only=True)
            if optimizer_artifact is not None
            else None
        )
    state_manager.import_model_state(model_state)
    optimizer, _ = runtime.get_optimizer_and_scheduler()
    sync_master_params = getattr(optimizer, "sync_master_params_from_model", None)
    if callable(sync_master_params):
        sync_master_params()
    if optim_state is not None:
        state_manager.import_optimizer_state(OptimizerState(state=optim_state))
    if trainer_state is not None:
        state_manager.import_trainer_state(
            TrainerState(
                step_context=trainer_state.get("step_context"),
                dataloader=trainer_state.get("dataloader"),
                plugin_states=trainer_state.get("plugin_states"),
                rng=RngState(
                    cpu=trainer_state["rng"]["cpu"],
                    cuda=trainer_state["rng"].get("cuda"),
                ),
            )
        )

    runtime._run_step_phase(RuntimePhase.POST_LOAD)
    distributed_barrier()


def load_checkpoint_manifest(checkpoint_dir: str | Path) -> CheckpointManifest:
    checkpoint_dir = Path(checkpoint_dir)
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
                entries=[_model_state_meta_from_dict(entry) for entry in rank_meta["entries"]],
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


def save_runtime_spec(checkpoint_root: str | Path, runtime_spec: dict[str, Any]) -> None:
    if not isinstance(runtime_spec, dict) or not runtime_spec:
        raise ValueError("runtime_spec must be a non-empty mapping")
    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    spec_path = root / "runtime_spec.json"
    if spec_path.exists():
        existing = load_runtime_spec(root)
        if existing != runtime_spec:
            raise ValueError(f"runtime spec already exists with different contents: {spec_path}")
        return
    tmp_path = root / "runtime_spec.json.tmp"
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(runtime_spec, f, indent=2)
        f.write("\n")
    tmp_path.replace(spec_path)


def load_runtime_spec(checkpoint: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    candidates = (checkpoint_path / "runtime_spec.json", checkpoint_path.parent / "runtime_spec.json")
    for spec_path in candidates:
        if not spec_path.is_file():
            continue
        with spec_path.open("r", encoding="utf-8") as f:
            runtime_spec = json.load(f)
        if not isinstance(runtime_spec, dict) or not runtime_spec:
            raise ValueError("runtime_spec.json must contain a non-empty mapping")
        return runtime_spec
    raise ValueError(
        f"runtime spec not found for {checkpoint_path}; "
        "expected runtime_spec.json in the checkpoint root"
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


def _load_model_state_for_rank(
    checkpoint_dir: Path,
    manifest: CheckpointManifest,
    rank_metadata: RankCheckpointMetadata,
    rank: int,
) -> dict[str, torch.Tensor]:
    loaded_states: dict[int, dict[str, torch.Tensor]] = {}
    merged_state: dict[str, torch.Tensor] = {}
    for entry in rank_metadata.entries:
        if entry.source_rank is None:
            raise ValueError(f"manifest missing model source rank for entry={entry.state_key}, rank={rank}")
        source_rank = entry.source_rank
        artifact = _find_artifact(manifest, kind="model", rank=source_rank)
        if artifact is None:
            raise ValueError(
                f"manifest missing model artifact for source rank={source_rank} "
                f"(used by rank={rank}, entry={entry.state_key})"
            )
        if source_rank not in loaded_states:
            loaded_states[source_rank] = torch.load(
                checkpoint_dir / artifact.path,
                map_location="cpu",
                weights_only=True,
            )
        source_state = loaded_states[source_rank]
        if entry.state_key not in source_state:
            raise ValueError(
                f"model state missing entry={entry.state_key} in source rank={source_rank} artifact"
            )
        merged_state[entry.state_key] = source_state[entry.state_key]
    return merged_state


def _validate_manifest_for_runtime(
    manifest: CheckpointManifest,
    runtime,
    *,
    weights_only: bool = False,
) -> None:
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
        if not weights_only and _find_artifact(manifest, kind="trainer", rank=rank_id) is None:
            raise ValueError(f"manifest missing trainer artifact for rank={rank_id}")
    _validate_unique_artifacts(manifest)

    for rank_meta in manifest.ranks:
        for entry in rank_meta.entries:
            if entry.source_rank is None:
                raise ValueError(
                    f"manifest missing model source rank for entry={entry.state_key}, rank={rank_meta.rank}"
                )
            if entry.source_rank < 0 or entry.source_rank >= manifest.world_size:
                raise ValueError(
                    f"manifest model source rank out of range: rank={rank_meta.rank}, "
                    f"entry={entry.state_key}, source_rank={entry.source_rank}"
                )
            if _find_artifact(manifest, kind="model", rank=entry.source_rank) is None:
                raise ValueError(
                    f"manifest missing model artifact for source rank={entry.source_rank} "
                    f"(used by rank={rank_meta.rank}, entry={entry.state_key})"
                )

    if weights_only:
        return

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
            runtime_source_rank = runtime.optimizer_checkpoint_rank(rank_id)
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


def _check_min_free_space(path: Path, min_free_gb: float) -> None:
    if min_free_gb < 0:
        raise ValueError(f"min_free_gb must be >= 0, got {min_free_gb}")
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"insufficient free space for checkpoint: path={path} free_gb={free_gb:.2f} required_gb={min_free_gb:.2f}"
        )
