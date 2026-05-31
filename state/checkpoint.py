from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
from state.state import (
    OptimizerState,
    ParamState,
    RngState,
    StateManager,
    TrainerState,
)
from utils.distributed import distributed_barrier


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


def save_sharded_checkpoint(
    state_manager: StateManager,
    path: str | Path,
    *,
    min_free_gb: float | None = None,
) -> None:
    checkpoint_dir = Path(path)
    if min_free_gb is not None:
        _check_min_free_space(checkpoint_dir.parent, min_free_gb)
    tmp_dir = checkpoint_dir.with_name(f"{checkpoint_dir.name}.tmp")
    rank = dist.get_rank() if dist.is_initialized() else 0
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(f"checkpoint already exists: {checkpoint_dir}")
    if rank == 0:
        if checkpoint_dir.exists():
            checkpoint_dir.rmdir()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.parent.mkdir(parents=True, exist_ok=True)
    distributed_barrier()
    _save_sharded_checkpoint_contents(state_manager, tmp_dir)
    distributed_barrier()
    if rank == 0:
        tmp_dir.rename(checkpoint_dir)
    distributed_barrier()


def _save_sharded_checkpoint_contents(state_manager: StateManager, checkpoint_dir: Path) -> None:
    runtime = state_manager.runtime
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    model_state, rank_entries = state_manager.export_model_state()
    optimizer_state = state_manager.export_optimizer_state()
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
    trainer_path = checkpoint_dir / f"trainer_rank_{rank}.pt"
    torch.save(asdict(state_manager.export_trainer_state()), trainer_path)
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

    distributed_barrier()


def load_sharded_checkpoint(state_manager: StateManager, path: str | Path) -> None:
    runtime = state_manager.runtime
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
    state_manager.import_model_state(model_state)
    if optim_state is not None:
        state_manager.import_optimizer_state(OptimizerState(state=optim_state))
    if trainer_state is not None:
        state_manager.import_trainer_state(
            TrainerState(
                step=int(trainer_state.get("step", 0)),
                microbatch_idx=int(trainer_state.get("microbatch_idx", 0)),
                consumed_tokens=trainer_state.get("consumed_tokens"),
                dataloader=trainer_state.get("dataloader"),
                plugin_states=trainer_state.get("plugin_states"),
                rng=RngState(
                    cpu=trainer_state["rng"]["cpu"],
                    cuda=trainer_state["rng"].get("cuda"),
                ),
            )
        )

    from runtime.core import RuntimePhase

    runtime._run_phase(RuntimePhase.POST_LOAD)
    distributed_barrier()


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


def _validate_manifest_for_runtime(manifest: CheckpointManifest, runtime) -> None:
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


def _check_min_free_space(path: Path, min_free_gb: float) -> None:
    if min_free_gb < 0:
        raise ValueError(f"min_free_gb must be >= 0, got {min_free_gb}")
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"insufficient free space for checkpoint: path={path} free_gb={free_gb:.2f} required_gb={min_free_gb:.2f}"
        )
