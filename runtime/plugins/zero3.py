from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import threading

import torch
import torch.distributed as dist
import torch.nn as nn

from runtime.buffer_allocator import allocate_buffer
from runtime.core import ParamRole, RuntimePhase
from runtime.mesh import MeshAxis
from runtime.plugin import PluginId, RuntimePlugin
from state.state import ParamState


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


class _ImmediateWork:
    def wait(self) -> None:
        return


@dataclass(frozen=True)
class _GroupContext:
    group: dist.ProcessGroup | None
    world_size: int
    rank: int


class _ExecDirection(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass
class _Bucket:
    module: nn.Module
    params: list[nn.Parameter]
    role: ParamRole
    group_context: _GroupContext
    param_shapes: list[torch.Size]
    param_numels: list[int]
    buffer_size: int
    local_param: nn.Parameter
    index: int
    logical_names: list[str]
    prev_bucket: "_Bucket | None" = None
    next_bucket: "_Bucket | None" = None
    pending_exec_reductions: int = 0
    post_reduction_handles: list[dist.Work] = field(default_factory=list)
    exec_states: list["_BucketExecState"] = field(default_factory=list)


@dataclass
class _BucketExecState:
    data_buffer: torch.Tensor
    grad_buffer: torch.Tensor
    shard_buffer: torch.Tensor
    fwd_handle: dist.Work | _ImmediateWork | None = None
    bwd_handle: dist.Work | _ImmediateWork | None = None
    grad_handle: dist.Work | _AllReduceShardWork | _ReduceScatterShardWork | _ImmediateWork | None = None
    pending: int = 0
    attached: bool = False
    backward_materialized: bool = False


class Zero3Plugin(RuntimePlugin):
    """FSDP-lite ZeRO-3 plugin with module-level parameter materialization."""

    def __init__(
        self,
        wrap_cls: set[type[nn.Module]] | None = None,
    ):
        super().__init__(
            id=PluginId.ZERO3,
            name="zero3",
            owns_optimizer=True,
            runs_after={PluginId.PP, PluginId.CP, PluginId.TP, PluginId.SP},
        )
        self.wrap_cls = set(wrap_cls or {nn.Linear})
        self.dp_group: dist.ProcessGroup | None = None
        self.world_size = 1
        self.rank = 0
        self.buckets: list[_Bucket] = []
        self._materialized_buffers: list[torch.Tensor] = []
        self.bucket_order_checked = False
        self._observed_forward_order: list[_Bucket] = []
        self._observed_forward_set: set[int] = set()
        self._first_bucket: _Bucket | None = None
        self._last_bucket: _Bucket | None = None
        self.expert_params: list[nn.Parameter] = []
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._post_reduction_thread: threading.Thread | None = None
        self._post_reduction_cond = threading.Condition()

    def bind(self, runtime) -> None:
        super().bind(runtime)

    def transform_model(self, model: nn.Module) -> nn.Module:
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

    def on_phase(self, phase: RuntimePhase) -> None:
        if phase == RuntimePhase.PRE_FORWARD:
            assert self.runtime is not None
            if self.bucket_order_checked and self._first_bucket is not None:
                self._prefetch_bucket(self._first_bucket, direction=_ExecDirection.FORWARD)
        elif phase == RuntimePhase.POST_FORWARD:
            self._finalize_bucket_order()
        elif phase == RuntimePhase.PRE_BACKWARD:
            assert self.runtime is not None
            context = self.runtime.state.step_context
            self._reset_buckets(
                grad_accum_start=context.accum_start,
                grad_accum_end=context.is_step_boundary,
            )
            if context.is_step_boundary and self._use_async_worker():
                self._start_post_reduction_worker()
            if self.bucket_order_checked and self._last_bucket is not None:
                self._prefetch_bucket(self._last_bucket, direction=_ExecDirection.BACKWARD)
        elif phase == RuntimePhase.POST_BACKWARD:
            assert self.runtime is not None
            if (
                self.runtime.state.step_context.is_step_boundary
                and not self._use_async_worker()
                and self.runtime._post_grad_reduction_callbacks
            ):
                self._fire_post_reductions_sync()
        elif phase == RuntimePhase.PRE_STEP:
            self._wait_grad_sync()

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

        self._reset_buckets(grad_accum_start=True, grad_accum_end=True)
        self._add_hooks()
        for bucket in self.buckets:
            self._free_full_params(bucket)

    def _make_bucket(self, index: int, module: nn.Module, params: list[nn.Parameter], logical_names: list[str]) -> _Bucket:
        role = self.runtime.get_param_role(params[0])
        if any(self.runtime.get_param_role(param) != role for param in params):
            raise ValueError(f"Zero3 bucket {index} mixes param roles, which is unsupported")
        group_context = self._group_context_for_role(role)
        dtype, device = params[0].dtype, params[0].device
        param_shapes = [param.shape for param in params]
        param_numels = [param.numel() for param in params]
        buffer_size = self._padded_len(sum(param_numels), group_context.world_size)
        full_param = torch.zeros(buffer_size, dtype=dtype, device=device)
        offset = 0
        for param in params:
            full_param[offset : offset + param.numel()].copy_(param.detach().view(-1))
            offset += param.numel()

        shard_len = buffer_size // group_context.world_size
        shard_start = group_context.rank * shard_len
        shard_end = (group_context.rank + 1) * shard_len
        local_param = nn.Parameter(full_param[shard_start:shard_end].clone())
        shard_buffer = torch.empty(shard_len, dtype=dtype, device=device)
        exec_state_count = self.runtime.plan.pp_schedule.microbatches
        exec_states = [
            _BucketExecState(
                data_buffer=torch.empty(buffer_size, dtype=dtype, device=device),
                grad_buffer=torch.empty(buffer_size, dtype=dtype, device=device),
                shard_buffer=shard_buffer.clone(),
            )
            for _ in range(exec_state_count)
        ]
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
            self._record_forward_bucket(bucket)
            self._materialize_full_params(bucket, direction=_ExecDirection.FORWARD)
            if self.bucket_order_checked and bucket.next_bucket is not None:
                self._prefetch_bucket(bucket.next_bucket, direction=_ExecDirection.FORWARD)

        return hook

    def _make_free_forward_hook(self, bucket: _Bucket):
        def hook(_module: nn.Module, _inputs, outputs) -> None:
            self._register_backward_output_hooks(bucket, outputs)
            self._free_full_params(bucket)
            self._exec_state(bucket).fwd_handle = None

        return hook

    def _make_attach_grad_hook(self, bucket: _Bucket):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            state = self._exec_state(bucket)
            if not state.attached:
                offset = 0
                for param, numel, shape in zip(bucket.params, bucket.param_numels, bucket.param_shapes):
                    param.grad = state.grad_buffer[offset : offset + numel].view(shape)
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
                state.grad_handle = self._reduce_scatter_avg(bucket, state)
                if self._use_async_worker():
                    with self._post_reduction_cond:
                        bucket.pending_exec_reductions -= 1
                        self._post_reduction_cond.notify_all()

        return hook

    def _prefetch_bucket(self, bucket: _Bucket, direction: _ExecDirection) -> None:
        state = self._exec_state(bucket)
        if bucket.group_context.group is None or bucket.group_context.world_size == 1:
            state.data_buffer.copy_(bucket.local_param.detach())
            immediate = _ImmediateWork()
            if direction == _ExecDirection.FORWARD:
                state.fwd_handle = immediate
                return
            if direction == _ExecDirection.BACKWARD:
                state.bwd_handle = immediate
                return
            raise ValueError(f"unknown ZeRO3 prefetch direction={direction}")
        if direction == _ExecDirection.FORWARD:
            if state.fwd_handle is not None:
                return
            state.fwd_handle = dist.all_gather_into_tensor(
                state.data_buffer,
                bucket.local_param.detach().contiguous(),
                group=bucket.group_context.group,
                async_op=True,
            )
            return
        if direction == _ExecDirection.BACKWARD:
            if state.bwd_handle is not None:
                return
            state.bwd_handle = dist.all_gather_into_tensor(
                state.data_buffer,
                bucket.local_param.detach().contiguous(),
                group=bucket.group_context.group,
                async_op=True,
            )
            return
        raise ValueError(f"unknown ZeRO3 prefetch direction={direction}")

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
                if bucket.prev_bucket is not None:
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
        if direction == _ExecDirection.FORWARD:
            if state.fwd_handle is None:
                self._prefetch_bucket(bucket, direction=_ExecDirection.FORWARD)
            assert state.fwd_handle is not None
            state.fwd_handle.wait()
            state.fwd_handle = None
        elif direction == _ExecDirection.BACKWARD:
            if state.bwd_handle is None:
                self._prefetch_bucket(bucket, direction=_ExecDirection.BACKWARD)
            assert state.bwd_handle is not None
            state.bwd_handle.wait()
            state.bwd_handle = None
        else:
            raise ValueError(f"unknown ZeRO3 materialize direction={direction}")
        self._bind_full_params(bucket, state.data_buffer)

    def _materialize_full_params_sync(self, bucket: _Bucket) -> None:
        buffer = allocate_buffer(
            key=f"zero3.materialize_sync.bucket{bucket.index}",
            shape=(bucket.buffer_size,),
            dtype=bucket.local_param.dtype,
            device=bucket.local_param.device,
        )
        if bucket.group_context.group is None or bucket.group_context.world_size == 1:
            buffer.copy_(bucket.local_param.detach())
        else:
            dist.all_gather_into_tensor(
                buffer,
                bucket.local_param.detach().contiguous(),
                group=bucket.group_context.group,
            )
        self._materialized_buffers.append(buffer)
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

    def _reduce_scatter_avg(
        self,
        bucket: _Bucket,
        state: _BucketExecState,
    ) -> dist.Work | _AllReduceShardWork | _ReduceScatterShardWork:
        assert self.runtime is not None
        if bucket.group_context.group is None or bucket.group_context.world_size == 1:
            bucket.local_param.grad.add_(state.grad_buffer[: bucket.local_param.numel()])
            return _ImmediateWork()
        if dist.get_backend(bucket.group_context.group) == "gloo":
            work = dist.all_reduce(state.grad_buffer, op=dist.ReduceOp.AVG, group=bucket.group_context.group, async_op=True)
            shard_len = bucket.local_param.numel()
            shard_start = bucket.group_context.rank * shard_len
            shard_end = (bucket.group_context.rank + 1) * shard_len
            return _AllReduceShardWork(work, state.grad_buffer, bucket.local_param.grad, shard_start, shard_end)
        work = dist.reduce_scatter_tensor(
            state.shard_buffer,
            state.grad_buffer,
            op=dist.ReduceOp.AVG,
            group=bucket.group_context.group,
            async_op=True,
        )
        return _ReduceScatterShardWork(work, state.shard_buffer, bucket.local_param.grad)

    def materialize_model(self) -> None:
        self._materialized_buffers.clear()
        for bucket in self.buckets:
            self._materialize_full_params_sync(bucket)

    def reshard_model(self) -> None:
        for bucket in self.buckets:
            self._free_full_params(bucket)
        self._materialized_buffers.clear()

    def override_param_state_dict(self) -> tuple[dict[str, torch.Tensor], list[ParamState]] | None:
        state = {}
        metadata = []
        for bucket in self.buckets:
            state_key = f"zero3_bucket_{bucket.index}"
            state[state_key] = bucket.local_param.detach().cpu().clone()
            metadata.append(
                ParamState(
                    state_key=state_key,
                    logical_names=bucket.logical_names,
                    logical_shapes=[tuple(shape) for shape in bucket.param_shapes],
                    physical_shape=tuple(bucket.local_param.shape),
                    dtype=str(bucket.local_param.dtype),
                )
            )
        return state, metadata

    def annotate_checkpoint_state(self, entry: ParamState) -> None:
        bucket_by_key = {f"zero3_bucket_{bucket.index}": bucket for bucket in self.buckets}
        bucket = bucket_by_key.get(entry.state_key)
        if bucket is None:
            return
        shard_len = bucket.local_param.numel()
        entry.set_plugin_annotation(
            self.id.value,
            {
                "bucket_index": bucket.index,
                "rank": bucket.group_context.rank,
                "world_size": bucket.group_context.world_size,
                "shard_offset": bucket.group_context.rank * shard_len,
                "shard_numel": shard_len,
                "numel": bucket.buffer_size,
            },
        )

    def load_param_state_dict(self, state: dict[str, torch.Tensor]) -> bool:
        for bucket in self.buckets:
            state_key = f"zero3_bucket_{bucket.index}"
            tensor = state[state_key].to(device=bucket.local_param.device, dtype=bucket.local_param.dtype)
            bucket.local_param.data.copy_(tensor)
            self._free_full_params(bucket)
        return True

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
            bucket.post_reduction_handles.clear()
            self._free_full_params(bucket)

    def _use_async_worker(self) -> bool:
        return self.dp_group is not None and dist.get_backend(self.dp_group) != "gloo"

    def _start_post_reduction_worker(self) -> None:
        assert self.runtime is not None
        if not self.runtime._post_grad_reduction_callbacks:
            self._post_reduction_thread = None
            return
        self._post_reduction_thread = threading.Thread(target=self._post_reduction_worker, daemon=True)
        self._post_reduction_thread.start()

    def _post_reduction_worker(self) -> None:
        assert self.runtime is not None
        callbacks = self.runtime._post_grad_reduction_callbacks
        for bucket in self.buckets:
            with self._post_reduction_cond:
                self._post_reduction_cond.wait_for(lambda: bucket.pending_exec_reductions == 0)
            for state in bucket.exec_states:
                if state.grad_handle is None:
                    raise RuntimeError("ZeRO3 bucket grad handle is None after backward")
                state.grad_handle.wait()
                state.grad_handle = None
            self._free_full_params(bucket)
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
            relevant = [(cb, rf) for cb, rf in callbacks if rf is None or rf == bucket.role]
            if not relevant:
                continue
            for state in bucket.exec_states:
                self._ensure_state_grad_handle(bucket, state)
                state.grad_handle.wait()
                state.grad_handle = None
            self._free_full_params(bucket)
            if bucket.local_param.grad is not None:
                for cb, _ in relevant:
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
                for state in bucket.exec_states:
                    self._ensure_state_grad_handle(bucket, state)
                    state.grad_handle.wait()
                    state.grad_handle = None
            self._free_full_params(bucket)

    def _flush_partial_grad_state_for_checkpoint(self) -> None:
        if self._post_reduction_thread is not None:
            self._post_reduction_thread.join()
            self._post_reduction_thread = None
        for bucket in self.buckets:
            for handle in bucket.post_reduction_handles:
                handle.wait()
            bucket.post_reduction_handles.clear()
            for state in bucket.exec_states:
                if state.grad_handle is not None:
                    state.grad_handle.wait()
                    state.grad_handle = None
                if state.bwd_handle is not None:
                    state.bwd_handle.wait()
                    state.bwd_handle = None
                if state.fwd_handle is not None:
                    state.fwd_handle.wait()
                    state.fwd_handle = None
            self._free_full_params(bucket)

    def _reset_buckets(self, *, grad_accum_start: bool, grad_accum_end: bool) -> None:
        self._materialized_buffers.clear()
        for bucket in self.buckets:
            if grad_accum_start:
                if bucket.local_param.grad is None:
                    bucket.local_param.grad = torch.zeros_like(bucket.local_param.data)
                else:
                    bucket.local_param.grad.zero_()
                bucket.post_reduction_handles.clear()
            bucket.pending_exec_reductions = len(bucket.exec_states) if grad_accum_end else 0
            for state in bucket.exec_states:
                if grad_accum_start:
                    state.grad_handle = None
            backward_state = self._exec_state(bucket)
            if backward_state.grad_handle is not None:
                backward_state.grad_handle.wait()
                backward_state.grad_handle = None
            backward_state.bwd_handle = None
            backward_state.grad_buffer.zero_()
            backward_state.shard_buffer.zero_()
            backward_state.pending = len(bucket.params)
            backward_state.attached = False
            backward_state.backward_materialized = False

    def _padded_len(self, numel: int, world_size: int) -> int:
        return (numel + world_size - 1) // world_size * world_size

    def _group_context_for_role(self, role: ParamRole) -> _GroupContext:
        assert self.runtime is not None
        if role == ParamRole.EXPERT:
            group = self.runtime.get_group(MeshAxis.EDP)
            if group is None:
                return _GroupContext(None, 1, 0)
            return _GroupContext(group, dist.get_world_size(group), dist.get_rank(group))
        assert self.dp_group is not None
        return _GroupContext(self.dp_group, self.world_size, self.rank)

    def _ensure_state_grad_handle(self, bucket: _Bucket, state: _BucketExecState) -> None:
        if state.grad_handle is not None:
            return
        if bucket.local_param.grad is None:
            bucket.local_param.grad = torch.zeros_like(bucket.local_param.data)
        state.grad_buffer.zero_()
        state.shard_buffer.zero_()
        state.grad_handle = self._reduce_scatter_avg(bucket, state)


    def _exec_state(self, bucket: _Bucket) -> _BucketExecState:
        if len(bucket.exec_states) == 1:
            return bucket.exec_states[0]
        assert self.runtime is not None
        context = self.runtime.state.step_context
        return bucket.exec_states[context.pp_cur_microbatch_idx]


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
