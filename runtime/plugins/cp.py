from __future__ import annotations

from typing import TYPE_CHECKING

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
from parallel.context_batch import ContextParallelBatchSharder
from parallel.context_token_planner import ContextTokenPlanner, build_context_token_planner
from runtime.layers.all_gather_attention import AllGatherKvAttentionCore
from runtime.layers.ring_attention import RingAttentionCore
from runtime.mesh import MeshAxis
from runtime.plugin import ContextParallelizableModule, PluginId, RuntimePlugin
from runtime.plugins.zero_common import ChainedWork
from runtime.types import ParamRole, RuntimePhase, SetupPhase
from utils.attention_backend import AttentionBackend
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
        self._token_planner: ContextTokenPlanner | None = None

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
        assert self.runtime is not None
        planner_type = self.runtime.plan.cp_token_planner
        if planner_type is None:
            raise ValueError("ContextParallelPlugin requires ParallelPlan.cp_token_planner")
        transformed_any = False
        for path in spec.attention_paths:
            if self.runtime.is_module_path_omitted(path):
                continue
            try:
                module = model.get_submodule(path)
            except AttributeError:
                raise
            _validate_supported_attention_module(module)
            attention_backend = getattr(module.attn_core, "attention_backend", AttentionBackend.EAGER)
            core = _build_cp_attention_core(
                self.cp_group,
                self.runtime.plan.cp_attn_core,
                step_context=self.runtime.state.step_context,
                attention_backend=attention_backend,
            )
            module.attn_core = core
            transformed_any = True
        if not transformed_any:
            raise ValueError("ContextParallelPlugin found no attention cores to transform")
        self._token_planner = build_context_token_planner(planner_type)
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
            if self.runtime.plan.batch_data_cp_aware:
                # The loader used this plugin's exact planner.  Do not slice a
                # second time; restore the canonical CP token contribution for
                # logging because the local batch contains only 1 / CP tokens.
                tokens = self.runtime.state.metadata.get("tokens")
                if isinstance(tokens, int):
                    self.runtime.state.metadata["tokens"] = tokens * self.world_size
            else:
                if self._token_planner is None:
                    raise RuntimeError("ContextParallelPlugin has no token planner after setup")
                self.runtime.state.batch = ContextParallelBatchSharder(
                    planner=self._token_planner,
                    world_size=self.world_size,
                ).shard(self.runtime.state.batch, rank=self.rank)
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
        if self.runtime.plan.cp_token_planner is None:
            raise ValueError("ContextParallelPlugin requires ParallelPlan.cp_token_planner")

def _build_cp_attention_core(
    group: dist.ProcessGroup,
    attention_core_type: ContextParallelAttentionCoreType,
    step_context: "StepContext | None" = None,
    attention_backend: str = AttentionBackend.EAGER,
) -> AllGatherKvAttentionCore | RingAttentionCore:
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
