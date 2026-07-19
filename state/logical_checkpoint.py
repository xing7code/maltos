from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from parallel.specs import TpSpShardAxis
from runtime.layers.moe import ExpertParallelMoE
from runtime.mesh import MeshAxis
from state.checkpoint import CheckpointManifest, load_checkpoint_manifest
from state.state import ModelStateMeta

if TYPE_CHECKING:
    from runtime.core import RuntimeCore


_LOGICAL_SINGLE_NAME = "model.safetensors"
_LOGICAL_INDEX_NAME = "model.safetensors.index.json"
_DEFAULT_LOGICAL_SHARD_SIZE_BYTES = 5 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class LogicalCheckpointManifest:
    metadata: dict[str, str | int]
    weight_map: dict[str, str]


def save_logical_checkpoint(
    path: str | Path,
    tensors: dict[str, torch.Tensor],
    *,
    max_shard_size_bytes: int = _DEFAULT_LOGICAL_SHARD_SIZE_BYTES,
) -> None:
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if max_shard_size_bytes < 1:
        raise ValueError(f"max_shard_size_bytes must be >= 1, got {max_shard_size_bytes}")
    cpu_tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in sorted(tensors.items())}
    shards = _plan_logical_shards(cpu_tensors, max_shard_size_bytes=max_shard_size_bytes)
    total_shards = len(shards)
    weight_map: dict[str, str] = {}
    for shard_idx, shard_tensors in enumerate(shards, start=1):
        shard_name = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
        save_file(shard_tensors, str(checkpoint_dir / shard_name))
        for name in shard_tensors:
            weight_map[name] = shard_name
    index = LogicalCheckpointManifest(
        metadata={
            "format": "logical_model",
            "version": 1,
            "total_size": sum(tensor.numel() * tensor.element_size() for tensor in cpu_tensors.values()),
        },
        weight_map=weight_map,
    )
    with (checkpoint_dir / _LOGICAL_INDEX_NAME).open("w", encoding="utf-8") as f:
        json.dump(asdict(index), f, indent=2, sort_keys=True)
        f.write("\n")


def load_logical_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint_dir = Path(path)
    index_path = checkpoint_dir / _LOGICAL_INDEX_NAME
    single_path = checkpoint_dir / _LOGICAL_SINGLE_NAME
    if index_path.is_file():
        manifest = _load_logical_index(index_path)
        tensors: dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(manifest.weight_map.values())):
            shard_tensors = load_file(str(checkpoint_dir / shard_name), device="cpu")
            for name, mapped_shard in manifest.weight_map.items():
                if mapped_shard == shard_name:
                    if name not in shard_tensors:
                        raise ValueError(f"logical checkpoint shard={shard_name} missing tensor={name}")
                    tensors[name] = shard_tensors[name]
        return tensors
    if single_path.is_file():
        return load_file(str(single_path), device="cpu")
    raise ValueError(
        f"logical checkpoint not found in {checkpoint_dir}; "
        f"expected {_LOGICAL_INDEX_NAME} or {_LOGICAL_SINGLE_NAME}"
    )


def load_logical_tensor(path: str | Path, name: str) -> torch.Tensor:
    checkpoint_dir = Path(path)
    index_path = checkpoint_dir / _LOGICAL_INDEX_NAME
    single_path = checkpoint_dir / _LOGICAL_SINGLE_NAME
    if index_path.is_file():
        manifest = _load_logical_index(index_path)
        shard_name = manifest.weight_map.get(name)
        if shard_name is None:
            raise KeyError(f"logical checkpoint missing tensor={name}")
        with safe_open(str(checkpoint_dir / shard_name), framework="pt", device="cpu") as f:
            return f.get_tensor(name)
    if single_path.is_file():
        with safe_open(str(single_path), framework="pt", device="cpu") as f:
            if name not in f.keys():
                raise KeyError(f"logical checkpoint missing tensor={name}")
            return f.get_tensor(name)
    raise ValueError(
        f"logical checkpoint not found in {checkpoint_dir}; "
        f"expected {_LOGICAL_INDEX_NAME} or {_LOGICAL_SINGLE_NAME}"
    )


def iter_logical_checkpoint_tensors(path: str | Path) -> Iterable[tuple[str, torch.Tensor]]:
    checkpoint_dir = Path(path)
    index_path = checkpoint_dir / _LOGICAL_INDEX_NAME
    single_path = checkpoint_dir / _LOGICAL_SINGLE_NAME
    if index_path.is_file():
        manifest = _load_logical_index(index_path)
        by_shard: dict[str, list[str]] = {}
        for name, shard_name in manifest.weight_map.items():
            by_shard.setdefault(shard_name, []).append(name)
        for shard_name in sorted(by_shard):
            with safe_open(str(checkpoint_dir / shard_name), framework="pt", device="cpu") as f:
                for name in sorted(by_shard[shard_name]):
                    yield name, f.get_tensor(name)
        return
    if single_path.is_file():
        with safe_open(str(single_path), framework="pt", device="cpu") as f:
            for name in sorted(f.keys()):
                yield name, f.get_tensor(name)
        return
    raise ValueError(
        f"logical checkpoint not found in {checkpoint_dir}; "
        f"expected {_LOGICAL_INDEX_NAME} or {_LOGICAL_SINGLE_NAME}"
    )


def iter_logical_tensors_from_runtime_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model,
    ep_size: int = 1,
    dp_size: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
) -> Iterable[tuple[str, torch.Tensor]]:
    checkpoint_dir = Path(checkpoint_dir)
    if any(isinstance(module, ExpertParallelMoE) for module in model.modules()):
        raise NotImplementedError("offline logical checkpoint conversion does not yet support ExpertParallelMoE models")

    manifest = load_checkpoint_manifest(checkpoint_dir)
    shard_rules = _tp_shard_rules(model)
    logical_state = model.state_dict()
    target_shapes = {name: tuple(tensor.shape) for name, tensor in logical_state.items()}
    target_dtypes = {name: tensor.dtype for name, tensor in logical_state.items()}

    pieces_by_name: dict[str, list[tuple[int, torch.Tensor]]] = {}
    owned_entries = _owned_model_entries(manifest)
    loaded_states: dict[int, dict[str, torch.Tensor]] = {}
    zero3_groups: dict[tuple[object, ...], list[tuple[int, ModelStateMeta]]] = {}
    for owner_rank, entry in owned_entries:
        if _is_zero3_entry(entry):
            _, pp_idx, cp_idx, tp_idx = _rank_coordinates(
                rank=owner_rank,
                dp_size=dp_size,
                pp_size=pp_size,
                cp_size=cp_size,
                tp_size=tp_size,
            )
            group_key = _zero3_group_key(
                entry,
                pp_idx=pp_idx,
                tp_idx=tp_idx,
                shard_rules=shard_rules,
            )
            zero3_groups.setdefault(group_key, []).append((owner_rank, entry))
            continue
        artifact = _find_model_artifact(manifest, owner_rank)
        if artifact is None:
            raise ValueError(f"manifest missing model artifact for rank={owner_rank}")
        if owner_rank not in loaded_states:
            loaded_states[owner_rank] = torch.load(
                checkpoint_dir / artifact.path,
                map_location="cpu",
                weights_only=True,
            )
        state = loaded_states[owner_rank]
        if entry.state_key not in state:
            raise ValueError(
                f"model state missing entry={entry.state_key} in source rank={owner_rank} artifact"
            )
        raw_tensor = state[entry.state_key]
        for logical_name, local_tensor in _split_entry_tensor(entry, raw_tensor):
            pieces_by_name.setdefault(logical_name, []).append((owner_rank, local_tensor.detach().cpu().contiguous()))

    for group_key, group_entries in zero3_groups.items():
        representative_rank, logical_pieces = _reconstruct_zero3_bucket_group(
            checkpoint_dir=checkpoint_dir,
            manifest=manifest,
            group_entries=group_entries,
            loaded_states=loaded_states,
            dp_size=dp_size,
            pp_size=pp_size,
            cp_size=cp_size,
            tp_size=tp_size,
        )
        for logical_name, tensor in logical_pieces:
            pieces_by_name.setdefault(logical_name, []).append((representative_rank, tensor))

    missing = sorted(set(target_shapes) - set(pieces_by_name))
    if missing:
        raise ValueError(f"runtime checkpoint missing logical tensors: {missing[:8]}{' ...' if len(missing) > 8 else ''}")

    for name in sorted(target_shapes):
        logical = _assemble_logical_tensor(
            name=name,
            pieces=pieces_by_name[name],
            shard_rules=shard_rules,
            target_shape=target_shapes[name],
        )
        yield name, logical.to(dtype=target_dtypes[name]).cpu().contiguous()


def _build_runtime_model_state_from_logical_tensors(
    runtime: "RuntimeCore",
    tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[ModelStateMeta]]:
    _validate_supported_runtime_for_import(runtime)
    rank = dist.get_rank() if dist.is_initialized() else 0
    rank_entries = _runtime_model_state_entries(runtime)
    shard_rules = _tp_shard_rules(runtime.model)
    tp_group = runtime.get_group(MeshAxis.TP)
    tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
    tp_world_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    zero3_plugin = _find_zero3_like_plugin(runtime)
    model_state: dict[str, torch.Tensor] = {}
    zero3_buckets = {
        f"zero3_bucket_{bucket.index}": bucket
        for bucket in getattr(zero3_plugin, "buckets", [])
    }
    for entry in rank_entries:
        if entry.source_rank != rank:
            continue
        if _is_zero3_entry(entry):
            bucket = zero3_buckets.get(entry.state_key)
            if bucket is None:
                raise ValueError(f"zero3 runtime bucket metadata missing for state_key={entry.state_key}")
            model_state[entry.state_key] = _build_zero3_bucket_tensor(
                bucket=bucket,
                tensors=tensors,
                shard_rules=shard_rules,
                tp_rank=tp_rank,
                tp_world_size=tp_world_size,
                expected_shape=entry.physical_shape,
            )
            continue
        model_state[entry.state_key] = _build_runtime_entry_tensor(
            runtime=runtime,
            entry=entry,
            tensors=tensors,
            shard_rules=shard_rules,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
    return model_state, rank_entries


def _runtime_model_state_entries(runtime: "RuntimeCore") -> list[ModelStateMeta]:
    zero3_plugin = _find_zero3_like_plugin(runtime)
    if zero3_plugin is None:
        entries = [entry.meta for entry in runtime.state_manager.params.values()]
        entries.extend(entry.meta for entry in runtime.state_manager.buffers.values())
        return entries
    rank = dist.get_rank() if dist.is_initialized() else 0
    entries = [
        ModelStateMeta(
            state_key=f"zero3_bucket_{bucket.index}",
            logical_names=list(bucket.logical_names),
            logical_shapes=[tuple(shape) for shape in bucket.param_shapes],
            physical_shape=tuple(bucket.local_param.shape),
            dtype=str(bucket.local_param.dtype),
            source_rank=rank,
        )
        for bucket in zero3_plugin.buckets
    ]
    entries.extend(entry.meta for entry in runtime.state_manager.buffers.values())
    return entries


def _build_runtime_entry_tensor(
    *,
    runtime: "RuntimeCore",
    entry: ModelStateMeta,
    tensors: dict[str, torch.Tensor],
    shard_rules: dict[str, str],
    tp_rank: int,
    tp_world_size: int,
) -> torch.Tensor:
    target_tensor = runtime.state_manager.get_model_tensor(entry.state_key)
    parts: list[torch.Tensor] = []
    for logical_name, logical_shape in zip(entry.logical_names, entry.logical_shapes, strict=True):
        if logical_name not in tensors:
            raise KeyError(f"logical checkpoint missing tensor={logical_name}")
        local_tensor = _local_tensor_for_runtime_tensor(
            name=logical_name,
            logical=tensors[logical_name],
            shard_rules=shard_rules,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        if len(entry.logical_names) == 1 and tuple(local_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(
                f"logical tensor shape mismatch for {logical_name}: expected local shape={tuple(target_tensor.shape)}, "
                f"got={tuple(local_tensor.shape)}"
            )
        if len(entry.logical_names) > 1 and int(local_tensor.numel()) != int(math.prod(logical_shape)):
            raise ValueError(
                f"logical tensor size mismatch for {logical_name}: expected local numel={int(math.prod(logical_shape))}, "
                f"got={int(local_tensor.numel())}"
            )
        parts.append(local_tensor.detach().cpu().contiguous())
    if len(parts) == 1:
        result = parts[0]
    else:
        result = torch.cat([part.reshape(-1) for part in parts], dim=0).view(entry.physical_shape)
    if tuple(result.shape) != tuple(entry.physical_shape):
        raise ValueError(
            f"runtime checkpoint tensor shape mismatch for {entry.state_key}: "
            f"expected={entry.physical_shape}, got={tuple(result.shape)}"
        )
    return result.to(dtype=target_tensor.dtype).cpu().contiguous()


def _load_logical_index(index_path: Path) -> LogicalCheckpointManifest:
    with index_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    metadata = raw.get("metadata")
    weight_map = raw.get("weight_map")
    if not isinstance(metadata, dict):
        raise ValueError(f"logical checkpoint index missing metadata: {index_path}")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"logical checkpoint index missing weight_map: {index_path}")
    fmt = metadata.get("format")
    version = int(metadata.get("version", 0))
    if fmt != "logical_model":
        raise ValueError(f"unsupported logical checkpoint format={fmt!r}")
    if version != 1:
        raise ValueError(f"unsupported logical checkpoint version={version}")
    return LogicalCheckpointManifest(
        metadata={str(key): value for key, value in metadata.items()},
        weight_map={str(name): str(shard_name) for name, shard_name in weight_map.items()},
    )


def _plan_logical_shards(
    tensors: dict[str, torch.Tensor],
    *,
    max_shard_size_bytes: int,
) -> list[dict[str, torch.Tensor]]:
    if not tensors:
        return [{}]
    shards: list[dict[str, torch.Tensor]] = []
    current_shard: dict[str, torch.Tensor] = {}
    current_size = 0
    for name, tensor in tensors.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_shard and current_size + tensor_size > max_shard_size_bytes:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor
        current_size += tensor_size
    if current_shard:
        shards.append(current_shard)
    return shards


def _validate_supported_runtime_for_import(runtime: "RuntimeCore") -> None:
    # Offline logical->runtime conversion can target runtimes that have already
    # been transformed for PP/EP as long as checkpoint metadata provides the
    # runtime->logical name mapping.
    return


def _owned_model_entries(manifest: CheckpointManifest) -> list[tuple[int, ModelStateMeta]]:
    owned: list[tuple[int, ModelStateMeta]] = []
    for rank_meta in manifest.ranks:
        for entry in rank_meta.entries:
            if entry.source_rank is None:
                raise ValueError(
                    f"manifest missing model source rank for entry={entry.state_key}, rank={rank_meta.rank}"
                )
            if entry.source_rank == rank_meta.rank:
                owned.append((rank_meta.rank, entry))
    return owned


def _is_zero3_entry(entry: ModelStateMeta) -> bool:
    return entry.state_key.startswith("zero3_bucket_")


def _zero3_group_key(
    entry: ModelStateMeta,
    *,
    pp_idx: int,
    tp_idx: int,
    shard_rules: dict[str, str],
) -> tuple[object, ...]:
    uses_tp_shards = any(shard_rules.get(name) in {TpSpShardAxis.PARAM_OUT, TpSpShardAxis.PARAM_IN} for name in entry.logical_names)
    return (
        "zero3",
        pp_idx,
        tp_idx if uses_tp_shards else None,
        tuple(entry.logical_names),
        tuple(entry.logical_shapes),
    )


def _rank_coordinates(*, rank: int, dp_size: int, pp_size: int, cp_size: int, tp_size: int) -> tuple[int, int, int, int]:
    tp_idx = rank % tp_size
    rank //= tp_size
    cp_idx = rank % cp_size
    rank //= cp_size
    pp_idx = rank % pp_size
    rank //= pp_size
    dp_idx = rank
    if not (0 <= dp_idx < dp_size):
        raise ValueError(f"rank={rank} out of range for mesh dp={dp_size}, pp={pp_size}, cp={cp_size}, tp={tp_size}")
    return dp_idx, pp_idx, cp_idx, tp_idx


def _reconstruct_zero3_bucket_group(
    *,
    checkpoint_dir: Path,
    manifest: CheckpointManifest,
    group_entries: list[tuple[int, ModelStateMeta]],
    loaded_states: dict[int, dict[str, torch.Tensor]],
    dp_size: int,
    pp_size: int,
    cp_size: int,
    tp_size: int,
) -> tuple[int, list[tuple[str, torch.Tensor]]]:
    exemplar = group_entries[0][1]
    ordered_ranks = sorted(
        owner_rank
        for owner_rank, _ in group_entries
    )
    shards: list[torch.Tensor] = []
    for owner_rank in ordered_ranks:
        artifact = _find_model_artifact(manifest, owner_rank)
        if artifact is None:
            raise ValueError(f"manifest missing model artifact for rank={owner_rank}")
        if owner_rank not in loaded_states:
            loaded_states[owner_rank] = torch.load(
                checkpoint_dir / artifact.path,
                map_location="cpu",
                weights_only=True,
            )
        state = loaded_states[owner_rank]
        if exemplar.state_key not in state:
            raise ValueError(
                f"model state missing entry={exemplar.state_key} in source rank={owner_rank} artifact"
            )
        shards.append(state[exemplar.state_key].detach().cpu().contiguous().view(-1))

    full_bucket = torch.cat(shards, dim=0)
    logical_pieces: list[tuple[str, torch.Tensor]] = []
    offset = 0
    for logical_name, logical_shape in zip(exemplar.logical_names, exemplar.logical_shapes, strict=True):
        numel = int(math.prod(logical_shape))
        chunk = full_bucket[offset : offset + numel]
        if chunk.numel() != numel:
            raise ValueError(
                f"zero3 bucket={exemplar.state_key} too small to unpack logical tensor={logical_name} "
                f"with shape={logical_shape}"
            )
        logical_pieces.append((logical_name, chunk.view(logical_shape)))
        offset += numel
    representative_rank = min(ordered_ranks)
    return representative_rank, logical_pieces


def _find_model_artifact(manifest: CheckpointManifest, rank: int):
    for artifact in manifest.artifacts:
        if artifact.kind == "model" and artifact.rank == rank:
            return artifact
    return None


def _split_entry_tensor(entry: ModelStateMeta, tensor: torch.Tensor) -> Iterable[tuple[str, torch.Tensor]]:
    if len(entry.logical_names) == 1:
        yield entry.logical_names[0], tensor
        return
    flat = tensor.contiguous().view(-1)
    offset = 0
    for logical_name, logical_shape in zip(entry.logical_names, entry.logical_shapes, strict=True):
        numel = int(math.prod(logical_shape))
        chunk = flat[offset : offset + numel]
        if chunk.numel() != numel:
            raise ValueError(
                f"checkpoint entry={entry.state_key} too small to unpack logical tensor={logical_name} "
                f"with shape={logical_shape}"
            )
        yield logical_name, chunk.view(logical_shape)
        offset += numel
    if offset != flat.numel():
        raise ValueError(
            f"checkpoint entry={entry.state_key} packed tensor size mismatch: consumed={offset}, total={flat.numel()}"
        )


def _assemble_logical_tensor(
    *,
    name: str,
    pieces: list[tuple[int, torch.Tensor]],
    shard_rules: dict[str, str],
    target_shape: tuple[int, ...],
) -> torch.Tensor:
    if len(pieces) == 1:
        tensor = pieces[0][1]
        if tuple(tensor.shape) == tuple(target_shape):
            return tensor
        if shard_rules.get(name) is None:
            raise ValueError(
                f"logical tensor={name} has local shape={tuple(tensor.shape)} but expected full shape={target_shape}"
            )

    ordered_pieces = [tensor for _, tensor in sorted(pieces, key=lambda item: item[0])]
    shard_axis = shard_rules.get(name)
    if shard_axis == TpSpShardAxis.PARAM_OUT:
        logical = torch.cat(ordered_pieces, dim=0)
    elif shard_axis == TpSpShardAxis.PARAM_IN:
        logical = torch.cat(ordered_pieces, dim=1)
    elif len(ordered_pieces) == 1:
        logical = ordered_pieces[0]
    else:
        if all(tuple(piece.shape) == tuple(target_shape) for piece in ordered_pieces):
            first = ordered_pieces[0]
            if all(torch.equal(first, piece) for piece in ordered_pieces[1:]):
                return first
        raise ValueError(
            f"logical tensor={name} has {len(ordered_pieces)} runtime pieces but no shard metadata to combine them"
        )
    if tuple(logical.shape) != tuple(target_shape):
        raise ValueError(
            f"logical tensor={name} shape mismatch after reconstruction: expected={target_shape}, got={tuple(logical.shape)}"
        )
    return logical


def _find_zero3_like_plugin(runtime: RuntimeCore):
    for plugin in runtime.plugins:
        if hasattr(plugin, "materialize_model") and hasattr(plugin, "reshard_model") and hasattr(plugin, "buckets"):
            return plugin
    return None


def _tp_shard_rules(model) -> dict[str, str]:
    if not hasattr(model, "tpsp_parallelize_spec"):
        return {}
    rules: dict[str, str] = {}
    for rule in model.tpsp_parallelize_spec().rules:
        if rule.shard_axis in (TpSpShardAxis.PARAM_OUT, TpSpShardAxis.PARAM_IN):
            rules[f"{rule.module_path}.weight"] = rule.shard_axis
            rules[f"{rule.module_path}.bias"] = rule.shard_axis
    return rules


def _local_tensor_for_runtime_tensor(
    *,
    name: str,
    logical: torch.Tensor,
    shard_rules: dict[str, str],
    tp_rank: int,
    tp_world_size: int,
) -> torch.Tensor:
    shard_axis = shard_rules.get(name)
    if shard_axis == TpSpShardAxis.PARAM_OUT:
        dim = logical.size(0) // tp_world_size
        start = tp_rank * dim
        end = (tp_rank + 1) * dim
        return logical[start:end]
    if shard_axis == TpSpShardAxis.PARAM_IN:
        dim = logical.size(1) // tp_world_size
        start = tp_rank * dim
        end = (tp_rank + 1) * dim
        return logical[:, start:end]
    return logical


def _build_zero3_bucket_tensor(
    *,
    bucket,
    tensors: dict[str, torch.Tensor],
    shard_rules: dict[str, str],
    tp_rank: int,
    tp_world_size: int,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    full_bucket = torch.zeros(bucket.buffer_size, dtype=bucket.local_param.dtype, device="cpu")
    offset = 0
    for name, numel, shape in zip(bucket.logical_names, bucket.param_numels, bucket.param_shapes, strict=True):
        if name not in tensors:
            raise KeyError(f"logical checkpoint missing tensor={name}")
        local_tensor = _local_tensor_for_runtime_tensor(
            name=name,
            logical=tensors[name],
            shard_rules=shard_rules,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
        )
        expected_local_shape = tuple(shape)
        if tuple(local_tensor.shape) != expected_local_shape:
            raise ValueError(
                f"logical tensor shape mismatch for {name}: expected local shape={expected_local_shape}, "
                f"got={tuple(local_tensor.shape)}"
            )
        full_bucket[offset : offset + numel].copy_(local_tensor.to(dtype=full_bucket.dtype).reshape(-1).cpu())
        offset += numel
    shard_len = int(math.prod(expected_shape))
    shard_start = bucket.group_context.rank * shard_len
    shard_end = shard_start + shard_len
    result = full_bucket[shard_start:shard_end].view(expected_shape).contiguous()
    if tuple(result.shape) != tuple(expected_shape):
        raise ValueError(
            f"zero3 runtime checkpoint tensor shape mismatch for bucket={bucket.index}: "
            f"expected={expected_shape}, got={tuple(result.shape)}"
        )
    return result
