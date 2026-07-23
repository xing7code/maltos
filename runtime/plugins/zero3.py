from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.distributed as dist
import torch.nn as nn

from runtime.buffer_allocator import BufferHandle, BufferPolicy, acquire_buffer, release_buffer
from runtime.mesh import MeshAxis
from runtime.plugin import PluginId
from runtime.plugins.zero_common import (
    ChainedWork,
    CompletedWork,
    GroupContext,
    NarrowShardWork,
    ZeroPluginBase,
)
from runtime.types import ParamRole, RuntimePhase, SetupPhase
from state.state import ModelStateMeta
from utils.distributed import all_gather_single, reduce_scatter_single
from utils.activation_checkpoint import is_activation_checkpoint_recompute


class _ExecDirection(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass
class _Bucket:
    module: nn.Module
    params: list[nn.Parameter]
    role: ParamRole
    group_context: GroupContext
    param_shapes: list[torch.Size]
    param_numels: list[int]
    buffer_size: int
    local_param: nn.Parameter
    index: int
    logical_names: list[str]
    prev_bucket: "_Bucket | None" = None
    next_bucket: "_Bucket | None" = None
    pending_exec_reductions: int = 0
    exec_states: list["_BucketExecState"] = field(default_factory=list)


@dataclass
class _BucketExecState:
    data_buffer: torch.Tensor | None = None
    grad_buffer: torch.Tensor | None = None
    grad_buffer_handle: BufferHandle | None = None
    shard_buffer: torch.Tensor | None = None
    shard_buffer_handle: BufferHandle | None = None
    fwd_handle: dist.Work | CompletedWork | None = None
    bwd_handle: dist.Work | CompletedWork | None = None
    grad_work: ChainedWork | None = None
    pending: int = 0
    attached: bool = False
    backward_materialized: bool = False


class Zero3Plugin(ZeroPluginBase):
    """FSDP-lite ZeRO-3 plugin with module-level parameter materialization."""

    def __init__(
        self,
        wrap_cls: set[type[nn.Module]] | None = None,
    ):
        super().__init__(
            id=PluginId.ZERO3,
            name="zero3",
            owns_optimizer=True,
            owns_model_state=True,
            runs_after={PluginId.PP, PluginId.CP, PluginId.TP, PluginId.SP},
        )
        self.wrap_cls = set(wrap_cls or {nn.Linear})
        self.buckets: list[_Bucket] = []
        self._materialized_buffer_handles: list[BufferHandle] = []
        self.bucket_order_checked = False
        self._observed_forward_order: list[_Bucket] = []
        self._observed_forward_set: set[int] = set()
        self._first_bucket: _Bucket | None = None
        self._last_bucket: _Bucket | None = None
        self.expert_params: list[nn.Parameter] = []

    def on_setup_phase(self, phase: SetupPhase, model: nn.Module) -> nn.Module:
        if phase == SetupPhase.MATERIALIZE:
            assert self.runtime is not None
            self.dp_group = self.runtime.get_group(MeshAxis.DP)
            if self.dp_group is None:
                raise ValueError("Zero3Plugin requires mesh.dp > 1")
            self.world_size = dist.get_world_size(self.dp_group)
            self.rank = dist.get_rank(self.dp_group)
            for cls in list(self.wrap_cls):
                self.wrap_cls.update(self.runtime.get_module_replacements(cls))
            self._prepare_buckets(model)
            optimizer_params = [bucket.local_param for bucket in self.buckets]
            self.optimizer = self.runtime.create_optimizer(optimizer_params)
            self.scheduler = self.runtime.create_scheduler(self.optimizer)
            return model
        if phase == SetupPhase.FINALIZE:
            self._add_hooks()
        return model

    def on_step_phase(self, phase: RuntimePhase) -> None:
        if phase == RuntimePhase.PRE_FORWARD:
            assert self.runtime is not None
            if (
                self.bucket_order_checked
                and self._first_bucket is not None
                and self._first_bucket.group_context.supports_async_overlap
            ):
                self._prefetch_bucket(self._first_bucket, direction=_ExecDirection.FORWARD)
        elif phase == RuntimePhase.POST_FORWARD:
            self._finalize_bucket_order()
        elif phase == RuntimePhase.PRE_BACKWARD:
            assert self.runtime is not None
            context = self.runtime.state.step_context
            self._reset_buckets(
                grad_accum_start=context.accum_start,
                backward_start=context.backward_start,
            )
            if (
                self.bucket_order_checked
                and self._last_bucket is not None
                and self._last_bucket.group_context.supports_async_overlap
            ):
                self._prefetch_bucket(self._last_bucket, direction=_ExecDirection.BACKWARD)
        elif phase == RuntimePhase.POST_BACKWARD:
            # See Zero1Plugin.on_phase: run_step() callers may read .grad once it
            # returns, so this must block here rather than defer to PRE_STEP.
            if self.runtime is not None and self.runtime.state.step_context.is_step_boundary:
                self._wait_grad_sync()
        elif phase == RuntimePhase.PRE_STEP:
            self._maybe_clip_local_shards()

    def annotate_param_metadata(self) -> None:
        super().annotate_param_metadata()
        assert self.runtime is not None
        logical_name_by_runtime = {
            fq_name: state.logical_names[0]
            for fq_name, state in self.runtime.state_manager.param_states.items()
            if len(state.logical_names) == 1
        }
        for bucket in self.buckets:
            bucket.logical_names = [
                logical_name_by_runtime.get(runtime_name, runtime_name)
                for runtime_name in bucket.logical_names
            ]

    def _prepare_buckets(self, model: nn.Module) -> None:
        visited: set[str] = set()
        param_to_name = {id(param): name for name, param in model.named_parameters()}
        covered_param_ids: set[int] = set()
        bucket_specs: list[tuple[nn.Module, list[nn.Parameter], list[str]]] = []
        for module_name, module in model.named_modules():
            if not isinstance(module, tuple(self.wrap_cls)):
                continue
            if any(module_name.startswith(parent + ".") for parent in visited):
                continue
            params = [param for param in module.parameters(recurse=True) if param.requires_grad]
            if not params:
                continue
            visited.add(module_name)
            logical_names = [param_to_name[id(param)] for param in params]
            bucket_specs.append((module, params, logical_names))
            covered_param_ids.update(id(param) for param in params)

        uncovered = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and id(param) not in covered_param_ids
        ]
        if uncovered:
            raise ValueError(
                "Zero3Plugin wrap_cls does not cover all trainable parameters. "
                f"Uncovered params: {uncovered}"
            )

        if not bucket_specs:
            return

        for index, (module, params, logical_names) in enumerate(bucket_specs):
            bucket = self._make_bucket(index, module, params, logical_names)
            self.buckets.append(bucket)

        self._reset_buckets(grad_accum_start=True)
        for bucket in self.buckets:
            self._free_full_params(bucket)

    def _bucket_local_sq(self, bucket: _Bucket) -> torch.Tensor:
        assert self.runtime is not None
        if bucket.local_param.grad is None:
            return torch.zeros((), dtype=torch.float64, device=bucket.local_param.device)
        shard_sq = torch.zeros((), dtype=torch.float64, device=bucket.local_param.grad.device)
        local_shard_start = bucket.group_context.rank * bucket.local_param.numel()
        local_shard_end = local_shard_start + bucket.local_param.numel()
        offset = 0
        for param, logical_name, param_numel in zip(bucket.params, bucket.logical_names, bucket.param_numels, strict=True):
            seg_start = offset
            seg_end = offset + param_numel
            overlap_start = max(seg_start, local_shard_start)
            overlap_end = min(seg_end, local_shard_end)
            if overlap_start < overlap_end:
                local_start = overlap_start - local_shard_start
                local_end = overlap_end - local_shard_start
                factor = self.runtime.grad_norm_replica_factor(param)
                shard_sq.add_(bucket.local_param.grad[local_start:local_end].detach().double().pow(2).sum() / float(factor))
            offset = seg_end
        return shard_sq

    def _make_bucket(self, index: int, module: nn.Module, params: list[nn.Parameter], logical_names: list[str]) -> _Bucket:
        role = self.runtime.get_param_role(params[0])
        if any(self.runtime.get_param_role(param) != role for param in params):
            raise ValueError(f"Zero3 bucket {index} mixes param roles, which is unsupported")
        group_context = self._group_context_for_role(role)
        assert self.runtime.device is not None
        device = torch.device(self.runtime.device)
        dtype = self.runtime.dtype or params[0].dtype
        param_shapes = [param.shape for param in params]
        param_numels = [param.numel() for param in params]
        buffer_size = self._padded_len(sum(param_numels), group_context.world_size)
        full_param = torch.zeros(buffer_size, dtype=params[0].dtype, device=params[0].device)
        offset = 0
        for param in params:
            full_param[offset : offset + param.numel()].copy_(param.detach().view(-1))
            offset += param.numel()

        shard_len = buffer_size // group_context.world_size
        shard_start = group_context.rank * shard_len
        shard_end = (group_context.rank + 1) * shard_len
        local_param = nn.Parameter(full_param[shard_start:shard_end].to(device=device, dtype=dtype).clone())
        exec_state_count = self.runtime.plan.pp_schedule.microbatches
        exec_states = [_BucketExecState() for _ in range(exec_state_count)]
        return _Bucket(
            module=module,
            params=params,
            role=role,
            group_context=group_context,
            param_shapes=param_shapes,
            param_numels=param_numels,
            buffer_size=buffer_size,
            local_param=local_param,
            index=index,
            logical_names=logical_names,
            exec_states=exec_states,
        )

    def _add_hooks(self) -> None:
        for bucket in self.buckets:
            bucket.module.register_forward_pre_hook(self._make_materialize_forward_hook(bucket))
            bucket.module.register_forward_hook(self._make_free_forward_hook(bucket))
            for param in bucket.params:
                param.register_hook(self._make_attach_grad_hook(bucket))
                param.register_post_accumulate_grad_hook(self._make_reduce_grad_hook(bucket))

    def _make_materialize_forward_hook(self, bucket: _Bucket):
        def hook(_module: nn.Module, _inputs) -> None:
            state = self._exec_state(bucket)
            if is_activation_checkpoint_recompute():
                # ``checkpoint(..., use_reentrant=False)`` runs the wrapped layer a
                # second time while autograd is already in backward.  The normal
                # forward post-hook released this bucket's buffer, so bind the
                # explicitly prefetched backward copy instead.  Do not infer this
                # from bwd_handle: the next real forward may also see a stale state.
                # A backward output hook may have materialized this bucket before
                # checkpoint asks for its recomputation, then its parameter-gradient
                # hook may already have released the buffer.  The flag records that
                # history; buffer ownership decides whether it is usable now.
                if not state.backward_materialized or state.data_buffer is None:
                    self._materialize_full_params(bucket, direction=_ExecDirection.BACKWARD)
                    state.backward_materialized = True
                return
            self._record_forward_bucket(bucket)
            self._materialize_full_params(bucket, direction=_ExecDirection.FORWARD)
            # gloo does not support concurrent ops on groups sharing the same rank pair;
            # skip prefetch here to avoid racing with EP alltoall in the current bucket's forward.
            if (
                self.bucket_order_checked
                and bucket.next_bucket is not None
                and bucket.group_context.supports_async_overlap
            ):
                self._prefetch_bucket(bucket.next_bucket, direction=_ExecDirection.FORWARD)

        return hook

    def _make_free_forward_hook(self, bucket: _Bucket):
        def hook(_module: nn.Module, _inputs, outputs) -> None:
            if is_activation_checkpoint_recompute():
                # Parameter-gradient hooks still need these full tensors after the
                # recomputed module returns.  They release the buffer on completion.
                return
            state = self._exec_state(bucket)
            self._register_backward_output_hooks(bucket, outputs)
            self._free_full_params(bucket)
            self._release_data_buffer(state)
            state.fwd_handle = None

        return hook

    def _make_attach_grad_hook(self, bucket: _Bucket):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            state = self._exec_state(bucket)
            if not state.attached:
                grad_buffer = self._ensure_grad_buffer(bucket, state)
                offset = 0
                for param, numel, shape in zip(bucket.params, bucket.param_numels, bucket.param_shapes):
                    param.grad = grad_buffer[offset : offset + numel].view(shape)
                    offset += numel
                state.attached = True
            return grad

        return hook

    def _make_reduce_grad_hook(self, bucket: _Bucket):
        def hook(_param: nn.Parameter) -> None:
            state = self._exec_state(bucket)
            state.pending -= 1
            if state.pending == 0:
                if bucket.local_param.grad is None:
                    bucket.local_param.grad = torch.empty_like(bucket.local_param.data)
                state.grad_work = self._new_state_grad_work(bucket, state)
                state.grad_work.fire()
                self._free_full_params(bucket)
                self._release_data_buffer(state)
                bucket.pending_exec_reductions -= 1

        return hook

    def _prefetch_bucket(self, bucket: _Bucket, direction: _ExecDirection) -> None:
        state = self._exec_state(bucket)
        data_buffer = self._ensure_data_buffer(bucket, state)
        is_fwd = direction == _ExecDirection.FORWARD
        if bucket.group_context.group is None or bucket.group_context.world_size == 1:
            data_buffer.copy_(bucket.local_param.detach())
            if is_fwd:
                state.fwd_handle = CompletedWork()
            else:
                state.bwd_handle = CompletedWork()
            return
        if is_fwd:
            if state.fwd_handle is not None:
                return
        else:
            if state.bwd_handle is not None:
                return
        handle = all_gather_single(
            data_buffer,
            bucket.local_param.detach().contiguous(),
            group=bucket.group_context.group,
            async_op=True,
        )
        if is_fwd:
            state.fwd_handle = handle
        else:
            state.bwd_handle = handle

    def _record_forward_bucket(self, bucket: _Bucket) -> None:
        if self.bucket_order_checked or bucket.index in self._observed_forward_set:
            return
        self._observed_forward_set.add(bucket.index)
        self._observed_forward_order.append(bucket)

    def _register_backward_output_hooks(self, bucket: _Bucket, outputs) -> None:
        tensors = list(_iter_tensors(outputs))
        if not tensors:
            return

        def hook(grad: torch.Tensor) -> torch.Tensor:
            state = self._exec_state(bucket)
            if not state.backward_materialized:
                self._materialize_full_params(bucket, direction=_ExecDirection.BACKWARD)
                # gloo does not support concurrent ops on groups sharing the same rank pair;
                # skip prefetch here to avoid racing with EP alltoall in the current bucket's backward.
                if (
                    self.bucket_order_checked
                    and bucket.prev_bucket is not None
                    and bucket.group_context.supports_async_overlap
                ):
                    self._prefetch_bucket(bucket.prev_bucket, direction=_ExecDirection.BACKWARD)
                state.backward_materialized = True
            return grad

        for tensor in tensors:
            if tensor.requires_grad:
                tensor.register_hook(hook)

    def _finalize_bucket_order(self) -> None:
        if self.bucket_order_checked or len(self._observed_forward_order) != len(self.buckets):
            return
        self._first_bucket = self._observed_forward_order[0]
        self._last_bucket = self._observed_forward_order[-1]
        for prev_bucket, next_bucket in zip(self._observed_forward_order, self._observed_forward_order[1:]):
            prev_bucket.next_bucket = next_bucket
            next_bucket.prev_bucket = prev_bucket
        self.bucket_order_checked = True

    def _materialize_full_params(self, bucket: _Bucket, direction: _ExecDirection) -> None:
        state = self._exec_state(bucket)
        is_fwd = direction == _ExecDirection.FORWARD
        handle = state.fwd_handle if is_fwd else state.bwd_handle
        if handle is None:
            self._prefetch_bucket(bucket, direction=direction)
            handle = state.fwd_handle if is_fwd else state.bwd_handle
        assert handle is not None
        handle.wait()
        if is_fwd:
            state.fwd_handle = None
        else:
            state.bwd_handle = None
        if state.data_buffer is None:
            raise RuntimeError("Zero3 materialized params require an allocated data buffer")
        self._bind_full_params(bucket, state.data_buffer)

    def _materialize_full_params_sync(self, bucket: _Bucket) -> None:
        handle = acquire_buffer(
            shape=(bucket.buffer_size,),
            dtype=bucket.local_param.dtype,
            device=bucket.local_param.device,
            policy=BufferPolicy.CACHEABLE,
        )
        buffer = handle.tensor
        if bucket.group_context.group is None or bucket.group_context.world_size == 1:
            buffer.copy_(bucket.local_param.detach())
        else:
            all_gather_single(
                buffer,
                bucket.local_param.detach().contiguous(),
                group=bucket.group_context.group,
            )
        self._materialized_buffer_handles.append(handle)
        self._bind_full_params(bucket, buffer)

    def _bind_full_params(self, bucket: _Bucket, buffer: torch.Tensor) -> None:
        offset = 0
        for param, numel, shape in zip(bucket.params, bucket.param_numels, bucket.param_shapes):
            param.data = buffer[offset : offset + numel].view(shape)
            offset += numel

    def _free_full_params(self, bucket: _Bucket) -> None:
        if bucket.group_context.world_size == 1:
            self._bind_full_params(bucket, bucket.local_param.data)
            return
        for param in bucket.params:
            param.data = bucket.local_param.data

    def _reduce_scatter_avg_functor(self, bucket: _Bucket, state: _BucketExecState):
        assert self.runtime is not None
        assert bucket.local_param.grad is not None
        grad_buffer = self._ensure_grad_buffer(bucket, state)
        shard_buffer = self._ensure_shard_buffer(bucket, state)

        def functor() -> dist.Work:
            if bucket.group_context.group is None or bucket.group_context.world_size == 1:
                shard = grad_buffer[: bucket.local_param.numel()]
                shard_buffer.copy_(
                    shard
                    if bucket.group_context.correction == 1.0
                    else shard * bucket.group_context.correction
                )
                return CompletedWork()
            if bucket.group_context.is_gloo:
                # gloo path uses SUM (not AVG), so correction must also fold in
                # the 1/world_size averaging factor AVG would otherwise apply.
                gloo_correction = bucket.group_context.correction / bucket.group_context.world_size
                if gloo_correction != 1.0:
                    grad_buffer.mul_(gloo_correction)
                work = dist.all_reduce(grad_buffer, op=dist.ReduceOp.SUM, group=bucket.group_context.group, async_op=True)
                shard_len = bucket.local_param.numel()
                shard_start = bucket.group_context.rank * shard_len
                shard_end = (bucket.group_context.rank + 1) * shard_len
                return NarrowShardWork(
                    work, grad_buffer, shard_buffer, shard_start, shard_end
                )
            if bucket.group_context.correction != 1.0:
                grad_buffer.mul_(bucket.group_context.correction)
            work = reduce_scatter_single(
                shard_buffer,
                grad_buffer,
                op=dist.ReduceOp.AVG,
                group=bucket.group_context.group,
                async_op=True,
            )
            return work

        return functor

    def materialize_model(self) -> None:
        for bucket in self.buckets:
            self._free_full_params(bucket)
        self._release_materialized_buffers()
        for bucket in self.buckets:
            self._materialize_full_params_sync(bucket)

    def reshard_model(self) -> None:
        for bucket in self.buckets:
            self._free_full_params(bucket)
        self._release_materialized_buffers()

    def export_model_state(self, state_manager) -> tuple[dict[str, torch.Tensor], list[ModelStateMeta]]:
        state = {}
        metadata = []
        rank = dist.get_rank() if dist.is_initialized() else 0
        for bucket in self.buckets:
            state_key = f"zero3_bucket_{bucket.index}"
            state[state_key] = bucket.local_param.detach().cpu().clone()
            metadata.append(
                ModelStateMeta(
                    state_key=state_key,
                    logical_names=bucket.logical_names,
                    logical_shapes=[tuple(shape) for shape in bucket.param_shapes],
                    physical_shape=tuple(bucket.local_param.shape),
                    dtype=str(bucket.local_param.dtype),
                    source_rank=rank,
                )
            )
        buffer_entries = [entry.meta for entry in state_manager.buffers.values()]
        for entry in buffer_entries:
            if entry.source_rank is None:
                raise RuntimeError(f"checkpoint source rank not resolved for buffer={entry.state_key}")
            if entry.source_rank != rank:
                continue
            buffer = state_manager.get_model_tensor(entry.state_key)
            state[entry.state_key] = buffer.detach().cpu().clone()
        metadata.extend(buffer_entries)
        return state, metadata

    def import_model_state(self, state_manager, state: dict[str, torch.Tensor]) -> None:
        for bucket in self.buckets:
            state_key = f"zero3_bucket_{bucket.index}"
            tensor = state[state_key].to(device=bucket.local_param.device, dtype=bucket.local_param.dtype)
            bucket.local_param.data.copy_(tensor)
            self._free_full_params(bucket)
        for fq_name in state_manager.buffer_states:
            if fq_name not in state:
                continue
            buffer = state_manager.get_model_tensor(fq_name)
            tensor = state[fq_name].to(device=buffer.device, dtype=buffer.dtype)
            buffer.data.copy_(tensor)

    def export_plugin_state(self) -> dict[str, object]:
        assert self.runtime is not None
        if self.runtime.state.step_context.microbatch_idx == 0:
            return {}
        self._flush_partial_grad_state_for_checkpoint()
        grads: dict[str, torch.Tensor] = {}
        for bucket in self.buckets:
            if bucket.local_param.grad is None:
                continue
            grads[str(bucket.index)] = bucket.local_param.grad.detach().cpu().clone()
        if not grads:
            return {}
        return {"bucket_local_grads": grads}

    def import_plugin_state(self, state: dict[str, object]) -> None:
        grad_state = state.get("bucket_local_grads")
        if not isinstance(grad_state, dict):
            return
        for bucket in self.buckets:
            tensor = grad_state.get(str(bucket.index))
            if not torch.is_tensor(tensor):
                bucket.local_param.grad = None
                continue
            bucket.local_param.grad = tensor.to(
                device=bucket.local_param.device,
                dtype=bucket.local_param.dtype,
            ).clone()
            self._free_full_params(bucket)

    def _wait_grad_sync(self) -> None:
        for bucket in self.buckets:
            for state in bucket.exec_states:
                self._ensure_state_grad_work(bucket, state)
                assert state.grad_work is not None
                self._wait_state_grad_work(bucket, state)
                state.grad_work = None
            self._free_full_params(bucket)
            self._release_bucket_data_buffers(bucket)

    def _flush_partial_grad_state_for_checkpoint(self) -> None:
        for bucket in self.buckets:
            for state in bucket.exec_states:
                if state.grad_work is not None:
                    self._wait_state_grad_work(bucket, state)
                    state.grad_work = None
                if state.bwd_handle is not None:
                    state.bwd_handle.wait()
                    state.bwd_handle = None
                if state.fwd_handle is not None:
                    state.fwd_handle.wait()
                    state.fwd_handle = None
            self._free_full_params(bucket)
            for state in bucket.exec_states:
                self._release_data_buffer(state)

    def _reset_buckets(self, *, grad_accum_start: bool, backward_start: bool = True) -> None:
        self._release_materialized_buffers()
        for bucket in self.buckets:
            if grad_accum_start:
                if bucket.local_param.grad is None:
                    bucket.local_param.grad = torch.zeros_like(bucket.local_param.data)
                else:
                    bucket.local_param.grad.zero_()
            # Per-micro-step reduction counter: re-armed at every backward phase
            # start, including the step-boundary micro-step (grad_accum > 1) where
            # grad_accum_start is False. See StepContext.backward_start.
            if backward_start:
                bucket.pending_exec_reductions = len(bucket.exec_states)
            for state in bucket.exec_states:
                if grad_accum_start:
                    state.grad_work = None
            backward_state = self._exec_state(bucket)
            if backward_state.grad_work is not None:
                self._wait_state_grad_work(bucket, backward_state)
                backward_state.grad_work = None
            backward_state.bwd_handle = None
            self._ensure_grad_buffer(bucket, backward_state).zero_()
            self._ensure_shard_buffer(bucket, backward_state).zero_()
            backward_state.pending = len(bucket.params)
            backward_state.attached = False
            backward_state.backward_materialized = False

    def _ensure_state_grad_work(self, bucket: _Bucket, state: _BucketExecState) -> None:
        if state.grad_work is not None:
            return
        if bucket.local_param.grad is None:
            bucket.local_param.grad = torch.zeros_like(bucket.local_param.data)
        # Zero buffers for exec_states whose backward hook never fired. _reset_buckets zeros
        # only the exec_state for the CURRENT pp_cur_microbatch_idx at each PRE_BACKWARD.
        # In AFAB with pp=2, exec_state[0] is not zeroed until the second PRE_BACKWARD, so
        # exec_state[1] (first backward) may still have torch.empty garbage when its hook does
        # not fire (e.g. EP expert not selected in that microbatch). Without this zero_(), the
        # uninitialized buffer propagates NaN through the reduce-scatter.
        self._ensure_grad_buffer(bucket, state).zero_()
        self._ensure_shard_buffer(bucket, state).zero_()
        state.grad_work = self._new_state_grad_work(bucket, state)
        state.grad_work.fire()

    def _new_state_grad_work(
        self, bucket: _Bucket, state: _BucketExecState
    ) -> ChainedWork:
        work = ChainedWork(
            None,
            self._reduce_scatter_avg_functor(bucket, state),
            blocks_by_stream=bucket.group_context.supports_async_overlap,
        )
        return self._apply_chained_work_wrappers(
            work, state.shard_buffer, bucket.role
        )

    def _wait_state_grad_work(
        self, bucket: _Bucket, state: _BucketExecState
    ) -> None:
        assert state.grad_work is not None
        assert bucket.local_param.grad is not None
        state.grad_work.wait()
        if state.shard_buffer is None:
            raise RuntimeError("Zero3 grad sync completed without an allocated shard buffer")
        bucket.local_param.grad.add_(state.shard_buffer)
        for param in bucket.params:
            param.grad = None
        state.attached = False
        self._release_grad_buffers(state)


    def _exec_state(self, bucket: _Bucket) -> _BucketExecState:
        if len(bucket.exec_states) == 1:
            return bucket.exec_states[0]
        assert self.runtime is not None
        context = self.runtime.state.step_context
        return bucket.exec_states[context.pp_cur_microbatch_idx]

    def _ensure_data_buffer(self, bucket: _Bucket, state: _BucketExecState) -> torch.Tensor:
        if state.data_buffer is None:
            state.data_buffer = torch.empty(
                bucket.buffer_size,
                dtype=bucket.local_param.dtype,
                device=bucket.local_param.device,
            )
        return state.data_buffer

    def _ensure_grad_buffer(self, bucket: _Bucket, state: _BucketExecState) -> torch.Tensor:
        if state.grad_buffer is None:
            handle = acquire_buffer(
                shape=(bucket.buffer_size,),
                dtype=bucket.local_param.dtype,
                device=bucket.local_param.device,
                policy=BufferPolicy.CACHEABLE,
            )
            state.grad_buffer_handle = handle
            state.grad_buffer = handle.tensor
        return state.grad_buffer

    def _ensure_shard_buffer(self, bucket: _Bucket, state: _BucketExecState) -> torch.Tensor:
        if state.shard_buffer is None:
            handle = acquire_buffer(
                shape=tuple(bucket.local_param.shape),
                dtype=bucket.local_param.dtype,
                device=bucket.local_param.device,
                policy=BufferPolicy.CACHEABLE,
            )
            state.shard_buffer_handle = handle
            state.shard_buffer = handle.tensor
        return state.shard_buffer

    def _release_data_buffer(self, state: _BucketExecState) -> None:
        state.data_buffer = None

    def _release_grad_buffers(self, state: _BucketExecState) -> None:
        if state.grad_buffer_handle is not None:
            release_buffer(state.grad_buffer_handle)
            state.grad_buffer_handle = None
        if state.shard_buffer_handle is not None:
            release_buffer(state.shard_buffer_handle)
            state.shard_buffer_handle = None
        state.grad_buffer = None
        state.shard_buffer = None

    def _release_bucket_data_buffers(self, bucket: _Bucket) -> None:
        for state in bucket.exec_states:
            self._release_data_buffer(state)

    def _release_materialized_buffers(self) -> None:
        for handle in self._materialized_buffer_handles:
            release_buffer(handle)
        self._materialized_buffer_handles.clear()


def _iter_tensors(obj):
    if torch.is_tensor(obj):
        yield obj
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_tensors(item)
        return
    if isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_tensors(item)
