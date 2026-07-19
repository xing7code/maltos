from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from runtime.core import RuntimeCore
    from runtime.types import StepContext

import torch
import torch.distributed as dist
import torch.nn as nn

from parallel.context_interfaces import (
    ContextParallelAttentionCore,
    ContextParallelAttentionCoreType,
)
from runtime.layers.all_gather_attention import AllGatherKvAttentionCore
from runtime.layers.ring_attention import RingAttentionCore
from runtime.mesh import MeshAxis
from runtime.plugin import ContextParallelizableModule, PluginId, RuntimePlugin
from runtime.plugins.zero_common import ChainedWork
from runtime.types import ParamRole, RuntimePhase, SetupPhase
from utils.attention_backend import AttentionBackend
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
from utils.distributed import all_reduce_tensor


class ContextParallelPlugin(RuntimePlugin):
    def __init__(self) -> None:
        super().__init__(
            id=PluginId.CP,
            name="context_parallel",
            runs_after={PluginId.TP, PluginId.SP, PluginId.DP},
        )
        self._grad_sync_handles: list[object] = []
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

    def bind(self, runtime: "RuntimeCore") -> None:
        super().bind(runtime)
        self._active_plugin_ids = {plugin.id for plugin in runtime.plugins if plugin is not self}
        active = self._active_plugin_ids
        self._use_param_hook_sync = (
            PluginId.DP not in active
            and PluginId.ZERO1 not in active
            and PluginId.ZERO2 not in active
            and PluginId.ZERO3 not in active
        )
        self._validate_runtime_support()

    def on_setup_phase(self, phase: SetupPhase, model: nn.Module) -> nn.Module:
        if phase == SetupPhase.TRANSFORM:
            return self._transform_attention_cores(model)
        if phase == SetupPhase.FINALIZE and self.world_size > 1 and self._use_param_hook_sync:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self._make_grad_sync_hook())
        return model

    def _transform_attention_cores(self, model: nn.Module) -> nn.Module:
        if self.world_size <= 1:
            return model
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
            attention_backend = getattr(module.attn_core, "attention_backend", AttentionBackend.EAGER)
            module.attn_core = _build_cp_attention_core(
                self.cp_group,
                self.runtime.plan.cp_attn_core,
                step_context=self.runtime.state.step_context,
                attention_backend=attention_backend,
            )
        return model

    def annotate_param_metadata(self) -> None:
        if self.world_size <= 1:
            return
        assert self.runtime is not None
        active = self._active_plugin_ids
        zero_active = bool({PluginId.ZERO1, PluginId.ZERO2, PluginId.ZERO3} & active)
        # Temporary lifecycle bridge: reduction-chain wiring must happen after
        # every plugin's setup transforms have run, because EP creates its grad
        # buckets there. annotate_param_metadata() is currently the first hook
        # with that guarantee. Keep the wiring isolated here so it can move
        # unchanged to a future post-transform/finalize hook.
        self._configure_expert_grad_sync_chain(zero_active=zero_active)
        if zero_active:
            return
        for fq_name in self.runtime.state_manager.param_states:
            param = self.runtime.state_manager.get_model_tensor(fq_name)
            if self.runtime.get_param_role(param) == ParamRole.EXPERT:
                continue
            attrs = self.runtime.state_manager.params[fq_name].attrs
            self.runtime.state_manager.update_model_state(
                fq_name,
                replicated_axes=attrs.replicated_axes | {MeshAxis.CP},
            )

    def _configure_expert_grad_sync_chain(self, *, zero_active: bool) -> None:
        """Append CP SUM when an EP reducer's EREP group excludes the CP axis."""
        assert self.runtime is not None
        active = self._active_plugin_ids
        # With reuse_cp_for_ep=True, EREP already spans the relevant CP sequence
        # shards and expert_erep_correction turns that part of AVG back into SUM.
        # When CP is not reused by EP, append an explicit CP SUM after EREP.
        reuse_cp = getattr(self.runtime.plan, "reuse_cp_for_ep", True)
        if PluginId.EP not in active or reuse_cp:
            return

        reducer_ids = {PluginId.ZERO1, PluginId.ZERO2, PluginId.ZERO3}
        if zero_active:
            reducer = next(
                plugin
                for plugin in self.runtime.plugins
                if plugin.id in reducer_ids
            )
        else:
            reducer = next(
                plugin
                for plugin in self.runtime.plugins
                if plugin.id == PluginId.EP
            )

        blocks_by_stream = dist.get_backend(self.cp_group) != "gloo"
        reducer.wrap_chained_work(
            lambda work, grad_buffer: ChainedWork(
                work,
                lambda: all_reduce_tensor(
                    grad_buffer,
                    op=dist.ReduceOp.SUM,
                    group=self.cp_group,
                    async_op=True,
                ),
                blocks_by_stream=blocks_by_stream,
            ),
            **({"role_filter": ParamRole.EXPERT} if zero_active else {}),
        )

    def on_step_phase(self, phase: RuntimePhase) -> None:
        if self.world_size <= 1:
            return
        if phase == RuntimePhase.PRE_STEP_RUNNER:
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
                all_reduce_tensor(param.grad, op=dist.ReduceOp.SUM, group=self.cp_group, async_op=True)
            )

        return hook

    def _maybe_launch_grad_sync(self) -> None:
        assert self.runtime is not None
        if self._use_param_hook_sync:
            return
        if not self.runtime.state.step_context.is_step_boundary:
            return
        if PluginId.ZERO1 in self._active_plugin_ids or PluginId.ZERO2 in self._active_plugin_ids or PluginId.ZERO3 in self._active_plugin_ids:
            return
        self._grad_sync_handles.clear()
        for param in self.runtime.model.parameters():
            if self.runtime.get_param_role(param) == ParamRole.EXPERT:
                continue
            if param.grad is None:
                continue
            self._grad_sync_handles.append(
                all_reduce_tensor(param.grad, op=dist.ReduceOp.SUM, group=self.cp_group, async_op=True)
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
    step_context: "StepContext | None" = None,
    attention_backend: str = AttentionBackend.EAGER,
) -> ContextParallelAttentionCore:
    if attention_core_type == ContextParallelAttentionCoreType.ALL_GATHER_KV:
        return AllGatherKvAttentionCore(group, attention_backend=attention_backend)
    if attention_core_type == ContextParallelAttentionCoreType.RING:
        return RingAttentionCore(group, step_context=step_context, attention_backend=attention_backend)
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
    if isinstance(batch, tuple):
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
    if isinstance(batch, list):
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
    raise TypeError(f"ContextParallelPlugin v0 does not support batch type={type(batch).__name__}")


def _infer_seq_len(batch: Any) -> int:
    if isinstance(batch, dict):
        return _infer_seq_len_from_dict(batch)
    if isinstance(batch, (tuple, list)):
        return _infer_seq_len_from_sequence(batch)
    raise TypeError(f"ContextParallelPlugin v0 does not support batch type={type(batch).__name__}")


def _infer_seq_len_from_dict(batch: dict[str, Any]) -> int:
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
    batch_size = int(reference.size(0))
    return positions.unsqueeze(0).expand(batch_size, -1).contiguous().to(device=device, dtype=torch.long)


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
    global_count = int((global_labels != IGNORE_INDEX).sum().item())
    local_count = int((local_labels != IGNORE_INDEX).sum().item())
    if global_count == 0:
        return None
    return local_count / global_count
