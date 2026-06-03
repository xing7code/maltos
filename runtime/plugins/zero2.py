from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn

from runtime.core import RuntimePhase
from runtime.mesh import MeshAxis
from runtime.plugin import PluginId, RuntimePlugin


class _AllReduceShardWork:
    def __init__(
        self,
        work: dist.Work,
        grad_buffer: torch.Tensor,
        local_grad: torch.Tensor,
        shard_start: int,
        shard_end: int,
    ):
        self.work = work
        self.grad_buffer = grad_buffer
        self.local_grad = local_grad
        self.shard_start = shard_start
        self.shard_end = shard_end

    def wait(self) -> None:
        self.work.wait()
        self.local_grad.add_(self.grad_buffer[self.shard_start : self.shard_end])


class _ReduceScatterShardWork:
    def __init__(
        self,
        work: dist.Work,
        shard_buffer: torch.Tensor,
        local_grad: torch.Tensor,
    ):
        self.work = work
        self.shard_buffer = shard_buffer
        self.local_grad = local_grad

    def wait(self) -> None:
        self.work.wait()
        self.local_grad.add_(self.shard_buffer)


@dataclass
class _Bucket:
    params: list[nn.Parameter]
    start: int
    end: int
    shard_start: int
    shard_end: int
    local_param: nn.Parameter
    exec_states: list["_BucketExecState"] = field(default_factory=list)


@dataclass
class _BucketExecState:
    grad_buffer: torch.Tensor
    shard_buffer: torch.Tensor
    pending: int = 0
    handle: dist.Work | _AllReduceShardWork | _ReduceScatterShardWork | None = None
    attached: bool = False


class Zero2Plugin(RuntimePlugin):
    """ZeRO-2 style optimizer and gradient sharding over the DP axis."""

    def __init__(
        self,
        bucket_mb_size: int = 25,
    ):
        super().__init__(id=PluginId.ZERO2, name="zero2", owns_optimizer=True)
        self.bucket_byte_size = bucket_mb_size * 1024 * 1024
        self.dp_group: dist.ProcessGroup | None = None
        self.world_size = 1
        self.rank = 0
        self.data_buffer: torch.Tensor | None = None
        self.buckets: list[_Bucket] = []
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    def transform_model(self, model: nn.Module) -> nn.Module:
        assert self.runtime is not None
        self.dp_group = self.runtime.get_group(MeshAxis.DP)
        if self.dp_group is None:
            raise ValueError("Zero2Plugin requires mesh.dp > 1")
        self.world_size = dist.get_world_size(self.dp_group)
        self.rank = dist.get_rank(self.dp_group)
        self._prepare_buffers_and_buckets(model)
        self.optimizer = self.runtime.create_optimizer([bucket.local_param for bucket in self.buckets])
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
        elif phase == RuntimePhase.POST_BACKWARD:
            assert self.runtime is not None
            if self.runtime.state.step_context.is_step_boundary:
                self._wait_last_grad_sync()
        elif phase == RuntimePhase.POST_STEP:
            self._gather_updated_params()
        elif phase == RuntimePhase.POST_LOAD:
            self._sync_local_params_from_data_buffer()

    def _prepare_buffers_and_buckets(self, model: nn.Module) -> None:
        params = [param for param in model.parameters() if param.requires_grad][::-1]
        if not params:
            return

        dtype, device = params[0].dtype, params[0].device
        param_buckets = self._build_param_buckets(params)
        padded_sizes = [self._padded_len(sum(param.numel() for param in bucket)) for bucket in param_buckets]
        self.data_buffer = torch.zeros(sum(padded_sizes), dtype=dtype, device=device)

        offset = 0
        for bucket_params, padded_size in zip(param_buckets, padded_sizes):
            param_offset = offset
            for param in bucket_params:
                with torch.no_grad():
                    self.data_buffer[param_offset : param_offset + param.numel()].copy_(param.detach().view(-1))
                    param.data = self.data_buffer[param_offset : param_offset + param.numel()].view_as(param)
                param_offset += param.numel()

            per_rank_size = padded_size // self.world_size
            shard_start = offset + self.rank * per_rank_size
            shard_end = offset + (self.rank + 1) * per_rank_size
            self.buckets.append(
                _Bucket(
                    params=bucket_params,
                    start=offset,
                    end=offset + padded_size,
                    shard_start=shard_start,
                    shard_end=shard_end,
                    local_param=nn.Parameter(self.data_buffer[shard_start:shard_end].clone()),
                    exec_states=[
                        _BucketExecState(
                            grad_buffer=torch.zeros(padded_size, dtype=dtype, device=device),
                            shard_buffer=torch.zeros(per_rank_size, dtype=dtype, device=device),
                        )
                        for _ in range(self.runtime.state.step_context.pp_num_microbatches)
                    ],
                )
            )
            offset += padded_size

        self._reset_buckets(grad_accum_start=True, grad_accum_end=True)
        self._add_param_hooks()

    def _build_param_buckets(self, params: list[nn.Parameter]) -> list[list[nn.Parameter]]:
        buckets: list[list[nn.Parameter]] = []
        curr_bucket: list[nn.Parameter] = []
        curr_bytes = 0
        for param in params:
            param_bytes = param.numel() * param.element_size()
            if curr_bucket and curr_bytes + param_bytes > self.bucket_byte_size:
                buckets.append(curr_bucket)
                curr_bucket = []
                curr_bytes = 0
            curr_bucket.append(param)
            curr_bytes += param_bytes
        if curr_bucket:
            buckets.append(curr_bucket)
        return buckets

    def _add_param_hooks(self) -> None:
        for bucket in self.buckets:
            for param in bucket.params:
                param.register_hook(self._make_attach_hook(bucket))
                param.register_post_accumulate_grad_hook(self._make_grad_sync_hook(bucket))

    def _make_attach_hook(self, bucket: _Bucket):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            state = self._exec_state(bucket)
            if not state.attached:
                self._attach_bucket_grad_buffer(bucket, state)
                state.attached = True
            return grad

        return hook

    def _attach_bucket_grad_buffer(self, bucket: _Bucket, state: _BucketExecState) -> None:
        offset = 0
        for param in bucket.params:
            param.grad = state.grad_buffer[offset : offset + param.numel()].view_as(param)
            offset += param.numel()

    def _make_grad_sync_hook(self, bucket: _Bucket):
        def hook(_param: nn.Parameter) -> None:
            state = self._exec_state(bucket)
            state.pending -= 1
            if state.pending == 0:
                if bucket.local_param.grad is None:
                    bucket.local_param.grad = torch.empty_like(bucket.local_param.data)
                state.handle = self._reduce_scatter_avg(bucket, state)

        return hook

    def _reduce_scatter_avg(
        self,
        bucket: _Bucket,
        state: _BucketExecState,
    ) -> dist.Work | _AllReduceShardWork | _ReduceScatterShardWork:
        assert self.dp_group is not None
        assert bucket.local_param.grad is not None
        bucket_numel = bucket.end - bucket.start
        full_grad = state.grad_buffer[:bucket_numel]
        if dist.get_backend(self.dp_group) == "gloo":
            work = dist.all_reduce(full_grad, op=dist.ReduceOp.AVG, group=self.dp_group, async_op=True)
            shard_len = bucket.local_param.numel()
            shard_start = self.rank * shard_len
            shard_end = (self.rank + 1) * shard_len
            return _AllReduceShardWork(work, full_grad, bucket.local_param.grad, shard_start, shard_end)
        work = dist.reduce_scatter_tensor(
            state.shard_buffer,
            full_grad,
            op=dist.ReduceOp.AVG,
            group=self.dp_group,
            async_op=True,
        )
        return _ReduceScatterShardWork(work, state.shard_buffer, bucket.local_param.grad)

    def _gather_updated_params(self) -> None:
        assert self.dp_group is not None
        assert self.data_buffer is not None
        handles = []
        with torch.no_grad():
            for bucket in self.buckets:
                handles.append(
                    dist.all_gather_into_tensor(
                        self.data_buffer[bucket.start : bucket.end],
                        bucket.local_param.detach().contiguous(),
                        group=self.dp_group,
                        async_op=True,
                    )
                )
            for handle in handles:
                handle.wait()

    def _wait_last_grad_sync(self) -> None:
        for bucket in self.buckets:
            waited = False
            for state in bucket.exec_states:
                if state.handle is None:
                    continue
                state.handle.wait()
                state.handle = None
                waited = True
            if not waited:
                raise RuntimeError("ZeRO2 bucket handle is None after backward")

    def _reset_buckets(self, *, grad_accum_start: bool, grad_accum_end: bool) -> None:
        if grad_accum_start:
            for bucket in self.buckets:
                if bucket.local_param.grad is None:
                    bucket.local_param.grad = torch.zeros_like(bucket.local_param.data)
                else:
                    bucket.local_param.grad.zero_()
        for bucket in self.buckets:
            state = self._exec_state(bucket)
            if state.handle is not None:
                state.handle.wait()
                state.handle = None
            state.grad_buffer.zero_()
            state.shard_buffer.zero_()
            state.pending = len(bucket.params)
            state.attached = False

    def _sync_local_params_from_data_buffer(self) -> None:
        assert self.data_buffer is not None
        with torch.no_grad():
            for bucket in self.buckets:
                bucket.local_param.data.copy_(self.data_buffer[bucket.shard_start : bucket.shard_end])

    def _padded_len(self, numel: int) -> int:
        return (numel + self.world_size - 1) // self.world_size * self.world_size

    def _exec_state(self, bucket: _Bucket) -> _BucketExecState:
        assert self.runtime is not None
        context = self.runtime.state.step_context
        return bucket.exec_states[context.pp_bwd_microbatch_idx]
