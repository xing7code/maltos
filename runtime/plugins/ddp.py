from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.distributed as dist

from runtime.buffer_allocator import BufferPolicy, acquire_buffer
from runtime.mesh import MeshAxis
from runtime.plugin import PluginId, RuntimePlugin
from runtime.types import ParamRole, RuntimePhase, SetupPhase


class _NullWork:
    def wait(self) -> None:
        pass


class DataParallelPlugin(RuntimePlugin):
    """RuntimeCore data parallel gradient synchronization."""

    def __init__(self, async_op: bool = False) -> None:
        name = "data_parallel_async" if async_op else "data_parallel"
        super().__init__(id=PluginId.DP, name=name)
        self.async_op = async_op

    def annotate_param_metadata(self) -> None:
        assert self.runtime is not None
        if self.runtime.mesh.dp <= 1:
            return
        for fq_name in self.runtime.state_manager.param_states:
            param = self.runtime.state_manager.get_model_tensor(fq_name)
            if self.runtime.get_param_role(param) == ParamRole.EXPERT:
                continue
            attrs = self.runtime.state_manager.params[fq_name].attrs
            self.runtime.state_manager.update_model_state(
                fq_name,
                replicated_axes=attrs.replicated_axes | {MeshAxis.DP},
            )

    def on_step_phase(self, phase: RuntimePhase) -> None:
        if phase != RuntimePhase.POST_BACKWARD:
            return
        assert self.runtime is not None
        if not self.runtime.state.step_context.is_step_boundary:
            return
        dp_group = self.runtime.get_group(MeshAxis.DP)
        if dp_group is None:
            return
        is_gloo = dist.get_backend(dp_group) == "gloo"
        world_size = dist.get_world_size(dp_group)
        handles = []
        for name, param in self.runtime.model.named_parameters():
            if not param.requires_grad:
                continue
            if self.runtime.get_param_role(param) == ParamRole.EXPERT:
                continue
            if param.grad is None:
                raise RuntimeError(f"param={name} requires grad but has no gradient")
            if is_gloo:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=dp_group)
                param.grad.div_(world_size)
            else:
                handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=dp_group, async_op=self.async_op)
                if self.async_op:
                    handles.append(handle)
        for handle in handles:
            handle.wait()

@dataclass
class _Bucket:
    group: dist.ProcessGroup
    params: list[nn.Parameter] = field(default_factory=list)
    param_views: list[torch.Tensor] = field(default_factory=list)
    flat_buffer: torch.Tensor | None = None
    total_bytes: int = 0
    pending: int = 0
    handle: dist.Work | None = None

    def add_param(self, param: nn.Parameter) -> None:
        self.params.append(param)
        self.total_bytes += param.numel() * param.element_size()

    def finalize(self) -> None:
        total_numel = sum(param.numel() for param in self.params)
        dtype, device = self.params[0].dtype, self.params[0].device
        self.flat_buffer = acquire_buffer(
            shape=(total_numel,),
            dtype=dtype,
            device=device,
            policy=BufferPolicy.PINNED,
            key=f"ddp.bucket.{id(self)}.flat",
        ).tensor
        self.flat_buffer.zero_()
        offset = 0
        for param in self.params:
            view = self.flat_buffer[offset : offset + param.numel()].view_as(param)
            self.param_views.append(view)
            offset += param.numel()
        self.reset(grad_accum_start=True, grad_accum_end=True)

    def reset(self, *, grad_accum_start: bool, grad_accum_end: bool) -> None:
        assert self.flat_buffer is not None
        self.pending = len(self.params) if grad_accum_end else 0
        self.handle = None
        if grad_accum_start:
            self.flat_buffer.zero_()
        for param, view in zip(self.params, self.param_views):
            param.grad = view

    def make_hook(self):
        is_gloo = dist.get_backend(self.group) == "gloo"
        world_size = dist.get_world_size(self.group)

        def hook(param):
            self.pending -= 1
            if self.pending == 0:
                assert self.flat_buffer is not None
                if is_gloo:
                    dist.all_reduce(self.flat_buffer, op=dist.ReduceOp.SUM, group=self.group)
                    self.flat_buffer.div_(world_size)
                    self.handle = _NullWork()
                else:
                    self.handle = dist.all_reduce(
                        self.flat_buffer,
                        op=dist.ReduceOp.AVG,
                        group=self.group,
                        async_op=True,
                    )

        return hook


class BucketDataParallelPlugin(RuntimePlugin):
    """Bucketized DDP that starts async grad sync during backward."""

    def __init__(self, bucket_mb_size: int = 25) -> None:
        super().__init__(id=PluginId.DP, name="bucket_data_parallel")
        self.bucket_byte_size = bucket_mb_size * 1024 * 1024
        self.buckets: list[_Bucket] = []

    def on_setup_phase(self, phase: SetupPhase, model: nn.Module) -> nn.Module:
        if phase == SetupPhase.MATERIALIZE:
            assert self.runtime is not None
            dp_group = self.runtime.get_group(MeshAxis.DP)
            if dp_group is None:
                return model
            self._build_buckets(model, dp_group)
            return model
        if phase == SetupPhase.FINALIZE:
            for bucket in self.buckets:
                for param in bucket.params:
                    param.register_post_accumulate_grad_hook(bucket.make_hook())
        return model

    def on_step_phase(self, phase: RuntimePhase) -> None:
        if phase == RuntimePhase.PRE_BACKWARD:
            assert self.runtime is not None
            context = self.runtime.state.step_context
            should_sync = context.is_step_boundary
            accum_start = context.accum_start
            for bucket in self.buckets:
                bucket.reset(grad_accum_start=accum_start, grad_accum_end=should_sync)
        elif phase == RuntimePhase.POST_BACKWARD:
            assert self.runtime is not None
            if not self.runtime.state.step_context.is_step_boundary:
                return
            for bucket in self.buckets:
                if bucket.handle is None:
                    raise RuntimeError(f"bucket with {len(bucket.params)} params was never synchronized")
                bucket.handle.wait()

    def annotate_param_metadata(self) -> None:
        assert self.runtime is not None
        if self.runtime.mesh.dp <= 1:
            return
        for fq_name in self.runtime.state_manager.param_states:
            param = self.runtime.state_manager.get_model_tensor(fq_name)
            if self.runtime.get_param_role(param) == ParamRole.EXPERT:
                continue
            attrs = self.runtime.state_manager.params[fq_name].attrs
            self.runtime.state_manager.update_model_state(
                fq_name,
                replicated_axes=attrs.replicated_axes | {MeshAxis.DP},
            )

    def _build_buckets(self, model: nn.Module, dp_group: dist.ProcessGroup) -> None:
        self.buckets = []
        current_bucket: _Bucket | None = None
        for param in reversed(
            [p for p in model.parameters() if p.requires_grad and self.runtime.get_param_role(p) != ParamRole.EXPERT]
        ):
            if current_bucket is None or current_bucket.total_bytes >= self.bucket_byte_size:
                if current_bucket is not None:
                    current_bucket.finalize()
                current_bucket = _Bucket(dp_group)
                self.buckets.append(current_bucket)
            current_bucket.add_param(param)
        if current_bucket is not None:
            current_bucket.finalize()
