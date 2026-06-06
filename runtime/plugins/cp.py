from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from parallel.context import (
    ContextParallelAttentionCore,
    ContextParallelAttentionCoreType,
)
from runtime.core import ParamRole, RuntimePhase
from runtime.mesh import MeshAxis
from runtime.plugin import ContextParallelizableModule, PluginId, RuntimePlugin
from runtime.plugins.cp_all_gather import AllGatherKvAttentionCore
from runtime.plugins.cp_ring import RingAttentionCore

# TODO: CP grad sync is currently deeply embedded in Zero1/2/3.
# We should revisit how to factor CP shard-gradient synchronization into a
# cleaner shared abstraction instead of teaching each ZeRO plugin its own
# CP-specific sync path.


class ContextParallelPlugin(RuntimePlugin):
    def __init__(self) -> None:
        super().__init__(
            id=PluginId.CP,
            name="context_parallel",
            runs_after={PluginId.TP, PluginId.SP, PluginId.DP},
        )
        self._grad_sync_handles: list[dist.Work] = []
        self._use_param_hook_sync = False

    @property
    def cp_group(self) -> dist.ProcessGroup:
        assert self.runtime is not None
        group = self.runtime.get_group(MeshAxis.CP)
        if group is None:
            raise ValueError("ContextParallelPlugin requires a CP process group")
        return group

    @property
    def rank(self) -> int:
        return dist.get_rank(self.cp_group)

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self.cp_group)

    def bind(self, runtime) -> None:
        super().bind(runtime)
        active = {plugin.id for plugin in runtime.plugins if plugin is not self}
        self._use_param_hook_sync = (
            PluginId.DP not in active
            and PluginId.ZERO1 not in active
            and PluginId.ZERO2 not in active
            and PluginId.ZERO3 not in active
        )
        self._validate_runtime_support()

    def transform_model(self, model: nn.Module) -> nn.Module:
        if self.world_size <= 1:
            return model
        if self._use_param_hook_sync:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self._make_grad_sync_hook())
        if not isinstance(model, ContextParallelizableModule):
            return model
        spec = model.context_parallel_spec()
        for path in spec.attention_paths:
            if self.runtime.is_module_path_omitted(path):
                continue
            try:
                module = model.get_submodule(path)
            except AttributeError:
                raise
            _validate_supported_attention_module(module)
            module.attn_core = _build_cp_attention_core(
                self.cp_group,
                self.runtime.plan.cp_attn_core,
            )
        return model

    def on_phase(self, phase: RuntimePhase) -> None:
        if self.world_size <= 1:
            return
        if phase == RuntimePhase.PRE_MICROBATCH:
            assert self.runtime is not None
            self.runtime.state.batch = _shard_batch_for_cp(
                self.runtime.state.batch,
                rank=self.rank,
                world_size=self.world_size,
                attention_core_type=self.runtime.plan.cp_attn_core,
            )
            return
        if phase == RuntimePhase.POST_BACKWARD:
            self._maybe_launch_grad_sync()
            return
        if phase == RuntimePhase.PRE_STEP:
            self._wait_grad_sync()

    def _make_grad_sync_hook(self):
        def hook(param: nn.Parameter) -> None:
            assert self.runtime is not None
            if not self.runtime.state.step_context.is_step_boundary:
                return
            if self.runtime.get_param_role(param) == ParamRole.EXPERT:
                return
            if param.grad is None:
                raise RuntimeError("ContextParallelPlugin expected param.grad before CP sync hook")
            self._grad_sync_handles.append(
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.cp_group, async_op=True)
            )

        return hook

    def _maybe_launch_grad_sync(self) -> None:
        assert self.runtime is not None
        if self._use_param_hook_sync:
            return
        if not self.runtime.state.step_context.is_step_boundary:
            return
        active = {plugin.id for plugin in self.runtime.plugins if plugin is not self}
        if PluginId.ZERO1 in active or PluginId.ZERO2 in active or PluginId.ZERO3 in active:
            return
        self._grad_sync_handles.clear()
        for param in self.runtime.model.parameters():
            if self.runtime.get_param_role(param) == ParamRole.EXPERT:
                continue
            if param.grad is None:
                continue
            self._grad_sync_handles.append(
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.cp_group, async_op=True)
            )

    def _wait_grad_sync(self) -> None:
        for handle in self._grad_sync_handles:
            handle.wait()
        self._grad_sync_handles.clear()

    def _validate_runtime_support(self) -> None:
        assert self.runtime is not None
        mesh = self.runtime.mesh
        if mesh.cp <= 1:
            raise ValueError("ContextParallelPlugin requires mesh.cp > 1")

def _build_cp_attention_core(
    group: dist.ProcessGroup,
    attention_core_type: ContextParallelAttentionCoreType,
) -> ContextParallelAttentionCore:
    if attention_core_type == ContextParallelAttentionCoreType.ALL_GATHER_KV:
        return AllGatherKvAttentionCore(group)
    if attention_core_type == ContextParallelAttentionCoreType.RING:
        return RingAttentionCore(group)
    raise ValueError(f"unsupported CP attention_core={attention_core_type!r}")


def _validate_supported_attention_module(module: nn.Module) -> None:
    attn_core = getattr(module, "attn_core", None)
    if not isinstance(attn_core, ContextParallelAttentionCore):
        raise TypeError(
            "ContextParallelPlugin requires attention modules to expose a protocol-compatible "
            f"`attn_core`, got module type={type(module).__name__}"
        )


def _shard_batch_for_cp(
    batch: Any,
    *,
    rank: int,
    world_size: int,
    attention_core_type: ContextParallelAttentionCoreType,
) -> Any:
    if world_size == 1:
        return batch
    seq_len = _infer_seq_len(batch)
    local_positions = _local_position_ids(
        seq_len,
        rank=rank,
        world_size=world_size,
        attention_core_type=attention_core_type,
    )
    if isinstance(batch, dict):
        sharded = dict(batch)
        for key in ("input_ids", "labels", "hidden_states", "position_ids"):
            value = sharded.get(key)
            sharded[key] = _shard_batch_item(value, local_positions, seq_len)
        sharded["position_ids"] = _materialize_position_ids(
            sharded.get("position_ids"),
            positions=local_positions,
            seq_len=seq_len,
            reference=_batch_reference_tensor(sharded),
        )
        sharded["position_offset"] = int(local_positions[0].item())
        sharded["loss_weight"] = _loss_weight(batch.get("labels"), sharded.get("labels"))
        return sharded
    if isinstance(batch, tuple):
        input_ids = _shard_batch_item(batch[0], local_positions, seq_len) if len(batch) > 0 else None
        labels = _shard_batch_item(batch[1], local_positions, seq_len) if len(batch) > 1 else None
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": _materialize_position_ids(
                None,
                positions=local_positions,
                seq_len=seq_len,
                reference=input_ids if torch.is_tensor(input_ids) else labels,
            ),
            "position_offset": int(local_positions[0].item()),
            "loss_weight": _loss_weight(batch[1] if len(batch) > 1 else None, labels),
        }
    if isinstance(batch, list):
        input_ids = _shard_batch_item(batch[0], local_positions, seq_len) if len(batch) > 0 else None
        labels = _shard_batch_item(batch[1], local_positions, seq_len) if len(batch) > 1 else None
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": _materialize_position_ids(
                None,
                positions=local_positions,
                seq_len=seq_len,
                reference=input_ids if torch.is_tensor(input_ids) else labels,
            ),
            "position_offset": int(local_positions[0].item()),
            "loss_weight": _loss_weight(batch[1] if len(batch) > 1 else None, labels),
        }
    raise TypeError(f"ContextParallelPlugin v0 does not support batch type={type(batch).__name__}")


def _infer_seq_len(batch: Any) -> int:
    if isinstance(batch, dict):
        return _infer_seq_len_from_dict(batch)
    if isinstance(batch, (tuple, list)):
        return _infer_seq_len_from_sequence(batch)
    raise TypeError(f"ContextParallelPlugin v0 does not support batch type={type(batch).__name__}")


def _infer_seq_len_from_dict(batch: dict[str, Any]) -> int:
    for key in ("input_ids", "labels", "hidden_states"):
        value = batch.get(key)
        if torch.is_tensor(value) and value.dim() >= 2:
            return int(value.size(1))
    position_ids = batch.get("position_ids")
    if torch.is_tensor(position_ids):
        if position_ids.dim() >= 2:
            return int(position_ids.size(1))
        if position_ids.dim() == 1:
            return int(position_ids.size(0))
    raise TypeError("ContextParallelPlugin could not infer sequence length from dict batch")


def _infer_seq_len_from_sequence(batch: tuple[Any, ...] | list[Any]) -> int:
    for value in batch:
        if torch.is_tensor(value) and value.dim() >= 2:
            return int(value.size(1))
    raise TypeError("ContextParallelPlugin could not infer sequence length from tuple/list batch")


def _local_seq_range(seq_len: int, rank: int, world_size: int) -> tuple[int, int]:
    if seq_len % world_size != 0:
        raise ValueError(
            "ContextParallelPlugin v0 requires sequence length divisible by cp world size, "
            f"got seq_len={seq_len}, cp={world_size}"
        )
    shard_len = seq_len // world_size
    return rank * shard_len, shard_len


def _local_position_ids(
    seq_len: int,
    *,
    rank: int,
    world_size: int,
    attention_core_type: ContextParallelAttentionCoreType,
) -> torch.Tensor:
    if attention_core_type != ContextParallelAttentionCoreType.RING:
        start, length = _local_seq_range(seq_len, rank, world_size)
        return torch.arange(start, start + length, dtype=torch.long)
    if seq_len % (2 * world_size) != 0:
        raise ValueError(
            "CP ring zigzag requires sequence length divisible by 2 * cp world size, "
            f"got seq_len={seq_len}, cp={world_size}"
        )
    half_len = seq_len // (2 * world_size)
    front_start = rank * half_len
    back_start = (2 * world_size - rank - 1) * half_len
    front = torch.arange(front_start, front_start + half_len, dtype=torch.long)
    back = torch.arange(back_start, back_start + half_len, dtype=torch.long)
    return torch.cat([front, back], dim=0)


def _batch_reference_tensor(batch: dict[str, Any]) -> torch.Tensor | None:
    for key in ("input_ids", "labels", "hidden_states"):
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
    if torch.is_tensor(current):
        sharded = _shard_batch_item(current, positions, seq_len)
        if torch.is_tensor(sharded):
            return sharded
    if reference is None:
        return positions.clone()
    batch_size = int(reference.size(0))
    return positions.unsqueeze(0).expand(batch_size, -1).contiguous()


def _shard_batch_item(value: Any, positions: torch.Tensor, seq_len: int) -> Any:
    if torch.is_tensor(value):
        index = positions.to(device=value.device)
        if value.dim() >= 2 and value.size(1) == seq_len:
            return value.index_select(1, index).contiguous()
        if value.dim() == 1 and value.size(0) == seq_len:
            return value.index_select(0, index).contiguous()
    return value


def _loss_weight(global_labels: Any, local_labels: Any) -> float | None:
    if not torch.is_tensor(global_labels) or not torch.is_tensor(local_labels):
        return None
    global_count = int((global_labels != -100).sum().item())
    local_count = int((local_labels != -100).sum().item())
    if global_count == 0:
        return None
    return local_count / global_count
