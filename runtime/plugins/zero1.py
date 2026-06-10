from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn

from runtime.core import ParamRole, RuntimePhase
from runtime.mesh import MeshAxis
from runtime.plugin import PluginId
from runtime.plugins.zero_common import GroupContext, _ZeroPluginBase


class _AllReduceShardWork:
    def __init__(
        self,
        work: dist.Work,
        grad_buffer: torch.Tensor,
        local_grad: torch.Tensor,
        shard_start: int,
        shard_end: int,
        correction: float = 1.0,
    ):
        self.work = work
        self.grad_buffer = grad_buffer
        self.local_grad = local_grad
        self.shard_start = shard_start
        self.shard_end = shard_end
        self.correction = correction

    def wait(self) -> None:
        self.work.wait()
        shard = self.grad_buffer[self.shard_start : self.shard_end]
        self.local_grad.copy_(shard if self.correction == 1.0 else shard * self.correction)


class _CorrectedWork:
    """Wraps a dist.Work and multiplies the output tensor by correction on wait()."""

    def __init__(self, work: dist.Work, tensor: torch.Tensor, correction: float) -> None:
        self.work = work
        self.tensor = tensor
        self.correction = correction

    def wait(self) -> None:
        self.work.wait()
        self.tensor.mul_(self.correction)


class _LocalCopyWork:
    def __init__(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        self.src = src
        self.dst = dst

    def wait(self) -> None:
        self.dst.copy_(self.src)


@dataclass
class _Bucket:
    params: list[nn.Parameter]
    group_context: GroupContext
    role: "ParamRole"
    start: int
    end: int
    shard_start: int
    shard_end: int
    local_param: nn.Parameter
    pending: int
    handle: dist.Work | _AllReduceShardWork | _LocalCopyWork | _CorrectedWork | None = None
    post_reduction_handles: list[dist.Work] = field(default_factory=list)


class Zero1Plugin(_ZeroPluginBase):
    """ZeRO-1 style optimizer-state sharding over the data-parallel axis."""

    def __init__(
        self,
        bucket_mb_size: int = 25,
    ):
        super().__init__(
            id=PluginId.ZERO1,
            name="zero1",
            owns_optimizer=True,
            runs_after={PluginId.PP, PluginId.CP, PluginId.TP, PluginId.SP},
            bucket_mb_size=bucket_mb_size,
        )
        self.data_buffer: torch.Tensor | None = None
        self.grad_buffer: torch.Tensor | None = None
        self.buckets: list[_Bucket] = []

    def transform_model(self, model: nn.Module) -> nn.Module:
        assert self.runtime is not None
        self.dp_group = self.runtime.get_group(MeshAxis.DP)
        if self.dp_group is None:
            raise ValueError("Zero1Plugin requires mesh.dp > 1")
        self.world_size = dist.get_world_size(self.dp_group)
        self.rank = dist.get_rank(self.dp_group)
        self._prepare_buffers_and_buckets(model)
        optimizer_params = [bucket.local_param for bucket in self.buckets]
        self.optimizer = self.runtime.create_optimizer(optimizer_params)
        self.scheduler = self.runtime.create_scheduler(self.optimizer)
        return model

    def on_phase(self, phase: RuntimePhase) -> None:
        if phase == RuntimePhase.PRE_BACKWARD:
            assert self.runtime is not None
            context = self.runtime.state.step_context
            self._reset_buckets(
                grad_accum_start=context.accum_start,
                grad_accum_end=context.is_step_boundary,
            )
            if context.is_step_boundary and self._use_async_worker():
                self._start_post_reduction_worker()
        elif phase == RuntimePhase.POST_BACKWARD:
            assert self.runtime is not None
            if (
                self.runtime.state.step_context.is_step_boundary
                and not self._use_async_worker()
            ):
                self._fire_post_reductions_sync()
        elif phase == RuntimePhase.PRE_STEP:
            self._wait_grad_sync()
        elif phase == RuntimePhase.POST_STEP:
            self._gather_updated_params()
        elif phase == RuntimePhase.POST_LOAD:
            self._sync_local_params_from_data_buffer()

    def _prepare_buffers_and_buckets(self, model: nn.Module) -> None:
        shared_params = self._role_params(model, ParamRole.SHARED)
        expert_params = self._role_params(model, ParamRole.EXPERT)
        bucket_specs: list[tuple[GroupContext, list[list[nn.Parameter]], ParamRole]] = []
        if shared_params:
            bucket_specs.append((self._group_context_for_role(ParamRole.SHARED), self._build_param_buckets(shared_params), ParamRole.SHARED))
        if expert_params:
            bucket_specs.append((self._group_context_for_role(ParamRole.EXPERT), self._build_param_buckets(expert_params), ParamRole.EXPERT))
        if not bucket_specs:
            return

        dtype = (shared_params or expert_params)[0].dtype
        device = (shared_params or expert_params)[0].device
        flat_specs: list[tuple[GroupContext, list[nn.Parameter], int, ParamRole]] = []
        for group_context, param_buckets, role in bucket_specs:
            for bucket_params in param_buckets:
                padded_size = self._padded_len(sum(param.numel() for param in bucket_params), group_context.world_size)
                flat_specs.append((group_context, bucket_params, padded_size, role))
        total_padded = sum(padded_size for _, _, padded_size, _ in flat_specs)
        self.data_buffer = torch.zeros(total_padded, dtype=dtype, device=device)
        self.grad_buffer = torch.zeros(total_padded, dtype=dtype, device=device)

        offset = 0
        for group_context, bucket_params, padded_size, role in flat_specs:
            param_offset = offset
            for param in bucket_params:
                with torch.no_grad():
                    self.data_buffer[param_offset : param_offset + param.numel()].copy_(param.detach().view(-1))
                    param.data = self.data_buffer[param_offset : param_offset + param.numel()].view_as(param)
                    param.grad = self.grad_buffer[param_offset : param_offset + param.numel()].view_as(param)
                param_offset += param.numel()

            per_rank_size = padded_size // group_context.world_size
            shard_start = offset + group_context.rank * per_rank_size
            shard_end = offset + (group_context.rank + 1) * per_rank_size
            self.buckets.append(
                _Bucket(
                    params=bucket_params,
                    group_context=group_context,
                    role=role,
                    start=offset,
                    end=offset + padded_size,
                    shard_start=shard_start,
                    shard_end=shard_end,
                    local_param=nn.Parameter(self.data_buffer[shard_start:shard_end].clone()),
                    pending=len(bucket_params),
                )
            )
            offset += padded_size

        self._reset_buckets(grad_accum_start=True, grad_accum_end=True)
        self._add_param_hooks()

    def _add_param_hooks(self) -> None:
        for bucket in self.buckets:
            for param in bucket.params:
                param.register_post_accumulate_grad_hook(self._make_grad_hook(bucket))

    def _make_grad_hook(self, bucket: _Bucket):
        def hook(_param: nn.Parameter) -> None:
            bucket.pending -= 1
            if bucket.pending == 0:
                if bucket.local_param.grad is None:
                    bucket.local_param.grad = torch.empty_like(bucket.local_param.data)
                bucket.handle = self._reduce_scatter_avg(bucket)
                if self._use_async_worker():
                    with self._post_reduction_cond:
                        self._post_reduction_cond.notify_all()

        return hook

    def _reduce_scatter_avg(self, bucket: _Bucket) -> dist.Work | _AllReduceShardWork | _LocalCopyWork | _CorrectedWork:
        assert self.grad_buffer is not None
        assert bucket.local_param.grad is not None
        full_grad = self.grad_buffer[bucket.start : bucket.end]
        if bucket.group_context.group is None or bucket.group_context.world_size == 1:
            return _LocalCopyWork(full_grad[bucket.shard_start - bucket.start : bucket.shard_end - bucket.start], bucket.local_param.grad)
        if bucket.group_context.is_gloo:
            work = dist.all_reduce(full_grad, op=dist.ReduceOp.AVG, group=bucket.group_context.group, async_op=True)
            shard_start = bucket.shard_start - bucket.start
            shard_end = bucket.shard_end - bucket.start
            return _AllReduceShardWork(work, full_grad, bucket.local_param.grad, shard_start, shard_end, bucket.group_context.correction)
        work = dist.reduce_scatter_tensor(
            bucket.local_param.grad,
            full_grad,
            op=dist.ReduceOp.AVG,
            group=bucket.group_context.group,
            async_op=True,
        )
        if bucket.group_context.correction == 1.0:
            return work
        return _CorrectedWork(work, bucket.local_param.grad, bucket.group_context.correction)

    def _gather_updated_params(self) -> None:
        assert self.data_buffer is not None
        assert self.grad_buffer is not None
        self.grad_buffer.zero_()
        handles = []
        with torch.no_grad():
            for bucket in self.buckets:
                local_data = bucket.local_param.detach().contiguous()
                if bucket.group_context.group is None or bucket.group_context.world_size == 1:
                    self.data_buffer[bucket.start : bucket.end].copy_(local_data)
                    continue
                handles.append(
                    dist.all_gather_into_tensor(
                        self.data_buffer[bucket.start : bucket.end],
                        local_data,
                        group=bucket.group_context.group,
                        async_op=True,
                    )
                )
            for handle in handles:
                handle.wait()

    def _post_reduction_worker(self) -> None:
        assert self.runtime is not None
        callbacks = self.runtime._post_grad_reduction_callbacks
        for bucket in self.buckets:
            with self._post_reduction_cond:
                self._post_reduction_cond.wait_for(lambda: bucket.handle is not None)
            if bucket.handle is None:
                raise RuntimeError("ZeRO1 bucket handle is None after backward")
            bucket.handle.wait()
            if bucket.local_param.grad is not None:
                for cb, role_filter in callbacks:
                    if role_filter is not None and bucket.role != role_filter:
                        continue
                    work = cb(bucket.local_param.grad)
                    if work is not None:
                        bucket.post_reduction_handles.append(work)

    def _fire_post_reductions_sync(self) -> None:
        assert self.runtime is not None
        callbacks = self.runtime._post_grad_reduction_callbacks
        for bucket in self.buckets:
            self._ensure_bucket_handle(bucket)
            if bucket.handle is None:
                raise RuntimeError("ZeRO1 bucket handle is None after backward")
            bucket.handle.wait()
            bucket.handle = None
            if bucket.local_param.grad is not None:
                for cb, role_filter in callbacks:
                    if role_filter is not None and bucket.role != role_filter:
                        continue
                    work = cb(bucket.local_param.grad)
                    if work is not None:
                        bucket.post_reduction_handles.append(work)

    def _wait_grad_sync(self) -> None:
        if self._post_reduction_thread is not None:
            self._post_reduction_thread.join()
            self._post_reduction_thread = None
        for bucket in self.buckets:
            if bucket.post_reduction_handles:
                for handle in bucket.post_reduction_handles:
                    handle.wait()
                bucket.post_reduction_handles.clear()
            else:
                self._ensure_bucket_handle(bucket)
                if bucket.handle is None:
                    continue
                bucket.handle.wait()
                bucket.handle = None

    def _reset_buckets(self, *, grad_accum_start: bool, grad_accum_end: bool) -> None:
        assert self.grad_buffer is not None
        if grad_accum_start:
            self.grad_buffer.zero_()
        for bucket in self.buckets:
            bucket.pending = len(bucket.params) if grad_accum_end else 0
            bucket.handle = None
            bucket.post_reduction_handles.clear()

    def _sync_local_params_from_data_buffer(self) -> None:
        assert self.data_buffer is not None
        with torch.no_grad():
            for bucket in self.buckets:
                bucket.local_param.data.copy_(self.data_buffer[bucket.shard_start : bucket.shard_end])

    def _ensure_bucket_handle(self, bucket: _Bucket) -> None:
        if bucket.handle is not None:
            return
        if bucket.local_param.grad is None:
            bucket.local_param.grad = torch.empty_like(bucket.local_param.data)
        bucket.handle = self._reduce_scatter_avg(bucket)
