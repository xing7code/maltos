from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn

from runtime.core import ParamRole, RuntimePhase
from runtime.mesh import MeshAxis
from runtime.plugin import PluginId
from runtime.plugins.zero_common import (
    ChainedWork,
    CompletedWork,
    GroupContext,
    NarrowShardWork,
    ZeroPluginBase,
)


@dataclass
class _Bucket:
    params: list[nn.Parameter]
    logical_names: list[str]
    group_context: GroupContext
    role: "ParamRole"
    start: int
    end: int
    shard_start: int
    shard_end: int
    local_param: nn.Parameter
    pending_exec_reductions: int = 0
    exec_states: list["_BucketExecState"] = field(default_factory=list)


@dataclass
class _BucketExecState:
    grad_buffer: torch.Tensor
    shard_buffer: torch.Tensor
    pending: int = 0
    work: ChainedWork | None = None
    attached: bool = False


class Zero2Plugin(ZeroPluginBase):
    """ZeRO-2 style optimizer and gradient sharding over the DP axis."""

    def __init__(
        self,
        bucket_mb_size: int = 25,
    ):
        super().__init__(
            id=PluginId.ZERO2,
            name="zero2",
            owns_optimizer=True,
            runs_after={PluginId.PP, PluginId.CP, PluginId.TP, PluginId.SP},
            bucket_mb_size=bucket_mb_size,
        )
        self.data_buffer: torch.Tensor | None = None
        self.buckets: list[_Bucket] = []

    def transform_model(self, model: nn.Module) -> nn.Module:
        assert self.runtime is not None
        self.dp_group = self.runtime.get_group(MeshAxis.DP)
        if self.dp_group is None:
            raise ValueError("Zero2Plugin requires mesh.dp > 1")
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
                backward_start=context.backward_start,
            )
        elif phase == RuntimePhase.POST_BACKWARD:
            # See Zero1Plugin.on_phase: run_step() callers may read .grad once it
            # returns, so this must block here rather than defer to PRE_STEP.
            if self.runtime is not None and self.runtime.state.step_context.is_step_boundary:
                self._wait_grad_sync()
        elif phase == RuntimePhase.PRE_STEP:
            self._maybe_clip_local_shards()
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
        param_to_name = {id(param): name for name, param in model.named_parameters()}
        flat_specs: list[tuple[GroupContext, list[nn.Parameter], int, ParamRole]] = []
        for group_context, param_buckets, role in bucket_specs:
            for bucket_params in param_buckets:
                padded_size = self._padded_len(sum(param.numel() for param in bucket_params), group_context.world_size)
                flat_specs.append((group_context, bucket_params, padded_size, role))
        self.data_buffer = torch.zeros(sum(padded_size for _, _, padded_size, _ in flat_specs), dtype=dtype, device=device)

        offset = 0
        for group_context, bucket_params, padded_size, role in flat_specs:
            param_offset = offset
            for param in bucket_params:
                with torch.no_grad():
                    self.data_buffer[param_offset : param_offset + param.numel()].copy_(param.detach().view(-1))
                    param.data = self.data_buffer[param_offset : param_offset + param.numel()].view_as(param)
                param_offset += param.numel()

            per_rank_size = padded_size // group_context.world_size
            shard_start = offset + group_context.rank * per_rank_size
            shard_end = offset + (group_context.rank + 1) * per_rank_size
            self.buckets.append(
                _Bucket(
                    params=bucket_params,
                    logical_names=[param_to_name[id(param)] for param in bucket_params],
                    group_context=group_context,
                    role=role,
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
                        for _ in range(self.runtime.plan.pp_schedule.microbatches)
                    ],
                )
            )
            offset += padded_size

        self._reset_buckets(grad_accum_start=True)
        self._add_param_hooks()

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
                state.work = self._new_state_work(bucket, state)
                state.work.fire()
                bucket.pending_exec_reductions -= 1

        return hook

    def _reduce_scatter_avg_functor(self, bucket: _Bucket, state: _BucketExecState):
        assert bucket.local_param.grad is not None

        def functor() -> dist.Work:
            bucket_numel = bucket.end - bucket.start
            full_grad = state.grad_buffer[:bucket_numel]
            if bucket.group_context.group is None or bucket.group_context.world_size == 1:
                shard = full_grad[: bucket.local_param.numel()]
                state.shard_buffer.copy_(shard)
                return CompletedWork()
            if bucket.group_context.is_gloo:
                # gloo path uses SUM (not AVG), so correction must also fold in
                # the 1/world_size averaging factor AVG would otherwise apply.
                gloo_correction = bucket.group_context.correction / bucket.group_context.world_size
                if gloo_correction != 1.0:
                    full_grad.mul_(gloo_correction)
                work = dist.all_reduce(full_grad, op=dist.ReduceOp.SUM, group=bucket.group_context.group, async_op=True)
                shard_len = bucket.local_param.numel()
                shard_start = bucket.group_context.rank * shard_len
                shard_end = (bucket.group_context.rank + 1) * shard_len
                return NarrowShardWork(
                    work, full_grad, state.shard_buffer, shard_start, shard_end
                )
            if bucket.group_context.correction != 1.0:
                full_grad.mul_(bucket.group_context.correction)
            work = dist.reduce_scatter_tensor(
                state.shard_buffer,
                full_grad,
                op=dist.ReduceOp.AVG,
                group=bucket.group_context.group,
                async_op=True,
            )
            return work

        return functor

    def _gather_updated_params(self) -> None:
        assert self.data_buffer is not None
        handles = []
        with torch.no_grad():
            for bucket in self.buckets:
                if bucket.group_context.group is None or bucket.group_context.world_size == 1:
                    self.data_buffer[bucket.start : bucket.end].copy_(bucket.local_param.detach())
                    continue
                handles.append(
                    dist.all_gather_into_tensor(
                        self.data_buffer[bucket.start : bucket.end],
                        bucket.local_param.detach().contiguous(),
                        group=bucket.group_context.group,
                        async_op=True,
                    )
                )
        for handle in handles:
            handle.wait()

    def _bucket_local_sq(self, bucket: _Bucket) -> torch.Tensor:
        assert self.runtime is not None
        if bucket.local_param.grad is None:
            return torch.zeros((), dtype=torch.float32, device=bucket.local_param.device)
        shard_sq = torch.zeros((), dtype=torch.float32, device=bucket.local_param.grad.device)
        local_shard_start = bucket.shard_start - bucket.start
        local_shard_end = bucket.shard_end - bucket.start
        offset = 0
        for logical_name, param in zip(bucket.logical_names, bucket.params, strict=True):
            seg_start = offset
            seg_end = offset + param.numel()
            overlap_start = max(seg_start, local_shard_start)
            overlap_end = min(seg_end, local_shard_end)
            if overlap_start < overlap_end:
                local_start = overlap_start - local_shard_start
                local_end = overlap_end - local_shard_start
                logical_param = self.runtime.state_manager.get_param_tensor(logical_name)
                factor = self.runtime.grad_norm_replica_factor(logical_param)
                shard_sq.add_(bucket.local_param.grad[local_start:local_end].detach().float().pow(2).sum() / float(factor))
            offset = seg_end
        return shard_sq

    def _wait_grad_sync(self) -> None:
        for bucket in self.buckets:
            waited = False
            for state in bucket.exec_states:
                self._ensure_state_work(bucket, state)
                assert state.work is not None
                self._wait_state_work(bucket, state)
                state.work = None
                waited = True
            if not waited:
                raise RuntimeError("ZeRO2 bucket has no exec_states after backward")

    def _reset_buckets(self, *, grad_accum_start: bool, backward_start: bool = True) -> None:
        if grad_accum_start:
            for bucket in self.buckets:
                if bucket.local_param.grad is None:
                    bucket.local_param.grad = torch.zeros_like(bucket.local_param.data)
                else:
                    bucket.local_param.grad.zero_()
        # Per-micro-step reduction counter: re-armed at every backward phase
        # start, including the step-boundary micro-step (grad_accum > 1) where
        # grad_accum_start is False. See StepContext.backward_start.
        if backward_start:
            for bucket in self.buckets:
                bucket.pending_exec_reductions = len(bucket.exec_states)
        for bucket in self.buckets:
            state = self._exec_state(bucket)
            if state.work is not None:
                self._wait_state_work(bucket, state)
                state.work = None
            state.grad_buffer.zero_()
            state.shard_buffer.zero_()
            state.pending = len(bucket.params)
            state.attached = False

    def _sync_local_params_from_data_buffer(self) -> None:
        assert self.data_buffer is not None
        with torch.no_grad():
            for bucket in self.buckets:
                bucket.local_param.data.copy_(self.data_buffer[bucket.shard_start : bucket.shard_end])

    def _ensure_state_work(self, bucket: _Bucket, state: _BucketExecState) -> None:
        if state.work is not None:
            return
        if bucket.local_param.grad is None:
            bucket.local_param.grad = torch.empty_like(bucket.local_param.data)
        # Do NOT zero grad_buffer here: _reset_buckets already zeroed it at PRE_BACKWARD.
        # Zeroing here would erase valid gradients when this is called after a completed
        # reduction (work was waited and set to None by _wait_grad_sync).
        state.work = self._new_state_work(bucket, state)
        state.work.fire()

    def _new_state_work(self, bucket: _Bucket, state: _BucketExecState) -> ChainedWork:
        work = ChainedWork(
            None,
            self._reduce_scatter_avg_functor(bucket, state),
            blocks_by_stream=bucket.group_context.supports_async_overlap,
        )
        return self._apply_chained_work_wrappers(work, state.shard_buffer, bucket.role)

    def _wait_state_work(self, bucket: _Bucket, state: _BucketExecState) -> None:
        assert state.work is not None
        assert bucket.local_param.grad is not None
        state.work.wait()
        bucket.local_param.grad.add_(state.shard_buffer)

    def _exec_state(self, bucket: _Bucket) -> _BucketExecState:
        if len(bucket.exec_states) == 1:
            return bucket.exec_states[0]
        assert self.runtime is not None
        context = self.runtime.state.step_context
        return bucket.exec_states[context.pp_cur_microbatch_idx]
