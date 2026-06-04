from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from parallel.context import (
    ContextParallelAttentionCore,
    ContextParallelAttentionCoreType,
)
from runtime.core import RuntimePhase
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
            try:
                module = model.get_submodule(path)
            except AttributeError:
                continue
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
        if mesh.ep != 1:
            raise ValueError(
                "ContextParallelPlugin v0 currently requires ep=1, "
                f"got dp={mesh.dp} pp={mesh.pp} cp={mesh.cp} tp={mesh.tp} ep={mesh.ep}"
            )
        active = {plugin.id for plugin in self.runtime.plugins if plugin is not self}
        unsupported = {PluginId.EP}
        overlap = sorted(plugin_id.value for plugin_id in active & unsupported)
        if overlap:
            raise ValueError(f"ContextParallelPlugin v0 does not yet support plugin combinations: {overlap}")

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


def _shard_batch_for_cp(batch: Any, *, rank: int, world_size: int) -> Any:
    if world_size == 1:
        return batch
    if isinstance(batch, dict):
        seq_len = _infer_seq_len_from_dict(batch)
        start, length = _local_seq_range(seq_len, rank, world_size)
        sharded = dict(batch)
        for key in ("input_ids", "labels", "hidden_states", "position_ids"):
            value = sharded.get(key)
            if torch.is_tensor(value) and value.dim() >= 2:
                sharded[key] = value.narrow(1, start, length).contiguous()
        sharded["position_offset"] = start
        sharded["loss_weight"] = _loss_weight(batch.get("labels"), sharded.get("labels"))
        return sharded
    if isinstance(batch, tuple):
        seq_len = _infer_seq_len_from_sequence(batch)
        start, length = _local_seq_range(seq_len, rank, world_size)
        input_ids = _shard_batch_item(batch[0], start, length) if len(batch) > 0 else None
        labels = _shard_batch_item(batch[1], start, length) if len(batch) > 1 else None
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_offset": start,
            "loss_weight": _loss_weight(batch[1] if len(batch) > 1 else None, labels),
        }
    if isinstance(batch, list):
        seq_len = _infer_seq_len_from_sequence(batch)
        start, length = _local_seq_range(seq_len, rank, world_size)
        input_ids = _shard_batch_item(batch[0], start, length) if len(batch) > 0 else None
        labels = _shard_batch_item(batch[1], start, length) if len(batch) > 1 else None
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_offset": start,
            "loss_weight": _loss_weight(batch[1] if len(batch) > 1 else None, labels),
        }
    raise TypeError(f"ContextParallelPlugin v0 does not support batch type={type(batch).__name__}")


def _infer_seq_len_from_dict(batch: dict[str, Any]) -> int:
    for key in ("input_ids", "labels", "hidden_states", "position_ids"):
        value = batch.get(key)
        if torch.is_tensor(value) and value.dim() >= 2:
            return int(value.size(1))
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


def _shard_batch_item(value: Any, start: int, length: int) -> Any:
    if torch.is_tensor(value) and value.dim() >= 2:
        return value.narrow(1, start, length).contiguous()
    return value


def _loss_weight(global_labels: Any, local_labels: Any) -> float | None:
    if not torch.is_tensor(global_labels) or not torch.is_tensor(local_labels):
        return None
    global_count = int((global_labels != -100).sum().item())
    local_count = int((local_labels != -100).sum().item())
    if global_count == 0:
        return None
    return local_count / global_count
