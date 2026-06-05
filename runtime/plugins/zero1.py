from __future__ import annotations

from dataclasses import dataclass
import threading

import torch
import torch.distributed as dist
import torch.nn as nn

from runtime.core import RuntimePhase
from runtime.mesh import MeshAxis
from runtime.plugin import PluginId, RuntimePlugin
from runtime.plugins.ep import is_ep_expert_param


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
        self.local_grad.copy_(self.grad_buffer[self.shard_start : self.shard_end])


@dataclass
class _Bucket:
    params: list[nn.Parameter]
    start: int
    end: int
    shard_start: int
    shard_end: int
    local_param: nn.Parameter
    pending: int
    handle: dist.Work | _AllReduceShardWork | None = None
    cp_handle: dist.Work | None = None


class Zero1Plugin(RuntimePlugin):
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
        )
        self.bucket_byte_size = bucket_mb_size * 1024 * 1024
        self.dp_group: dist.ProcessGroup | None = None
        self.cp_group: dist.ProcessGroup | None = None
        self.world_size = 1
        self.rank = 0
        self.data_buffer: torch.Tensor | None = None
        self.grad_buffer: torch.Tensor | None = None
        self.buckets: list[_Bucket] = []
        self.expert_params: list[nn.Parameter] = []
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._cp_sync_thread: threading.Thread | None = None
        self._cp_sync_cond = threading.Condition()

    def transform_model(self, model: nn.Module) -> nn.Module:
        assert self.runtime is not None
        self.dp_group = self.runtime.get_group(MeshAxis.DP)
        self.cp_group = self.runtime.get_group(MeshAxis.CP)
        if self.dp_group is None:
            raise ValueError("Zero1Plugin requires mesh.dp > 1")
        self.world_size = dist.get_world_size(self.dp_group)
        self.rank = dist.get_rank(self.dp_group)
        self.expert_params = [param for param in model.parameters() if param.requires_grad and is_ep_expert_param(self.runtime, param)]
        self._prepare_buffers_and_buckets(model)
        optimizer_params = [bucket.local_param for bucket in self.buckets] + self.expert_params
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
            if context.is_step_boundary and self._use_cp_sync_worker():
                self._maybe_start_cp_sync_worker()
        elif phase == RuntimePhase.POST_BACKWARD:
            assert self.runtime is not None
            if (
                self.runtime.state.step_context.is_step_boundary
                and self.cp_group is not None
                and dist.get_world_size(self.cp_group) > 1
                and not self._use_cp_sync_worker()
            ):
                self._launch_cp_grad_sync()
        elif phase == RuntimePhase.PRE_STEP:
            self._wait_grad_sync()
        elif phase == RuntimePhase.POST_STEP:
            self._gather_updated_params()
        elif phase == RuntimePhase.POST_LOAD:
            self._sync_local_params_from_data_buffer()

    def _prepare_buffers_and_buckets(self, model: nn.Module) -> None:
        params = [param for param in model.parameters() if param.requires_grad and not is_ep_expert_param(self.runtime, param)][::-1]
        if not params:
            return

        dtype, device = params[0].dtype, params[0].device
        param_buckets: list[list[nn.Parameter]] = []
        curr_bucket: list[nn.Parameter] = []
        curr_bytes = 0
        for param in params:
            param_bytes = param.numel() * param.element_size()
            if curr_bucket and curr_bytes + param_bytes > self.bucket_byte_size:
                param_buckets.append(curr_bucket)
                curr_bucket = []
                curr_bytes = 0
            curr_bucket.append(param)
            curr_bytes += param_bytes
        if curr_bucket:
            param_buckets.append(curr_bucket)

        padded_sizes = [self._padded_len(sum(param.numel() for param in bucket)) for bucket in param_buckets]
        self.data_buffer = torch.zeros(sum(padded_sizes), dtype=dtype, device=device)
        self.grad_buffer = torch.zeros(sum(padded_sizes), dtype=dtype, device=device)

        offset = 0
        for bucket_params, padded_size in zip(param_buckets, padded_sizes):
            param_offset = offset
            for param in bucket_params:
                with torch.no_grad():
                    self.data_buffer[param_offset : param_offset + param.numel()].copy_(param.detach().view(-1))
                    param.data = self.data_buffer[param_offset : param_offset + param.numel()].view_as(param)
                    param.grad = self.grad_buffer[param_offset : param_offset + param.numel()].view_as(param)
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
                if self._use_cp_sync_worker():
                    with self._cp_sync_cond:
                        self._cp_sync_cond.notify_all()

        return hook

    def _reduce_scatter_avg(self, bucket: _Bucket) -> dist.Work | _AllReduceShardWork:
        assert self.dp_group is not None
        assert self.grad_buffer is not None
        assert bucket.local_param.grad is not None
        full_grad = self.grad_buffer[bucket.start : bucket.end]
        if dist.get_backend(self.dp_group) == "gloo":
            work = dist.all_reduce(full_grad, op=dist.ReduceOp.AVG, group=self.dp_group, async_op=True)
            shard_start = bucket.shard_start - bucket.start
            shard_end = bucket.shard_end - bucket.start
            return _AllReduceShardWork(work, full_grad, bucket.local_param.grad, shard_start, shard_end)
        return dist.reduce_scatter_tensor(
            bucket.local_param.grad,
            full_grad,
            op=dist.ReduceOp.AVG,
            group=self.dp_group,
            async_op=True,
        )

    def _gather_updated_params(self) -> None:
        assert self.dp_group is not None
        assert self.data_buffer is not None
        assert self.grad_buffer is not None
        self.grad_buffer.zero_()
        handles = []
        with torch.no_grad():
            for bucket in self.buckets:
                local_data = bucket.local_param.detach().contiguous()
                handles.append(
                    dist.all_gather_into_tensor(
                        self.data_buffer[bucket.start : bucket.end],
                        local_data,
                        group=self.dp_group,
                        async_op=True,
                    )
                )
            for handle in handles:
                handle.wait()

    def _maybe_start_cp_sync_worker(self) -> None:
        if self.cp_group is None or dist.get_world_size(self.cp_group) <= 1:
            self._cp_sync_thread = None
            return
        self._cp_sync_thread = threading.Thread(target=self._cp_sync_worker, daemon=True)
        self._cp_sync_thread.start()

    def _cp_sync_worker(self) -> None:
        assert self.cp_group is not None
        for bucket in self.buckets:
            with self._cp_sync_cond:
                self._cp_sync_cond.wait_for(lambda: bucket.handle is not None)
            if bucket.handle is None:
                raise RuntimeError("ZeRO1 bucket handle is None after backward")
            bucket.handle.wait()
            bucket.handle = None
            if bucket.local_param.grad is None:
                continue
            bucket.cp_handle = dist.all_reduce(
                bucket.local_param.grad,
                op=dist.ReduceOp.SUM,
                group=self.cp_group,
                async_op=True,
            )

    def _launch_cp_grad_sync(self) -> None:
        for bucket in self.buckets:
            if bucket.handle is None:
                raise RuntimeError("ZeRO1 bucket handle is None after backward")
            bucket.handle.wait()
            bucket.handle = None
            if self.cp_group is None or dist.get_world_size(self.cp_group) <= 1:
                continue
            if bucket.local_param.grad is None:
                continue
            bucket.cp_handle = dist.all_reduce(
                bucket.local_param.grad,
                op=dist.ReduceOp.SUM,
                group=self.cp_group,
                async_op=True,
            )

    def _wait_grad_sync(self) -> None:
        if self._cp_sync_thread is not None:
            self._cp_sync_thread.join()
            self._cp_sync_thread = None
        for bucket in self.buckets:
            if bucket.cp_handle is None:
                if bucket.handle is None:
                    raise RuntimeError("ZeRO1 bucket handle is None after backward")
                bucket.handle.wait()
                bucket.handle = None
                continue
            bucket.cp_handle.wait()
            bucket.cp_handle = None

    def _use_cp_sync_worker(self) -> bool:
        return self.dp_group is not None and dist.get_backend(self.dp_group) != "gloo"

    def _reset_buckets(self, *, grad_accum_start: bool, grad_accum_end: bool) -> None:
        assert self.grad_buffer is not None
        if grad_accum_start:
            self.grad_buffer.zero_()
        for bucket in self.buckets:
            bucket.pending = len(bucket.params) if grad_accum_end else 0
            bucket.handle = None
            bucket.cp_handle = None

    def _sync_local_params_from_data_buffer(self) -> None:
        assert self.data_buffer is not None
        with torch.no_grad():
            for bucket in self.buckets:
                bucket.local_param.data.copy_(self.data_buffer[bucket.shard_start : bucket.shard_end])

    def _padded_len(self, numel: int) -> int:
        return (numel + self.world_size - 1) // self.world_size * self.world_size
