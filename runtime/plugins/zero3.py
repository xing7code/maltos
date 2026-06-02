from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from runtime.core import RuntimePhase
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
        self.local_grad.copy_(self.grad_buffer[self.shard_start : self.shard_end])


@dataclass
class _Bucket:
    module: nn.Module
    params: list[nn.Parameter]
    param_shapes: list[torch.Size]
    param_numels: list[int]
    buffer_size: int
    local_param: nn.Parameter
    data_buffer: torch.Tensor
    grad_buffer: torch.Tensor
    index: int
    logical_names: list[str]
    pending: int
    prev_bucket: "_Bucket | None" = None
    next_bucket: "_Bucket | None" = None
    fwd_handle: dist.Work | None = None
    bwd_handle: dist.Work | None = None
    attached: bool = False
    grad_handle: dist.Work | _AllReduceShardWork | None = None


class Zero3Plugin(RuntimePlugin):
    """FSDP-lite ZeRO-3 plugin with module-level parameter materialization."""

    def __init__(
        self,
        wrap_cls: set[type[nn.Module]] | None = None,
        enable_prefetch: bool = True,
    ):
        super().__init__(id=PluginId.ZERO3, name="zero3", owns_optimizer=True)
        self.wrap_cls = wrap_cls or {nn.Linear}
        self.enable_prefetch = enable_prefetch
        self.dp_group: dist.ProcessGroup | None = None
        self.world_size = 1
        self.rank = 0
        self.buckets: list[_Bucket] = []
        self.data_buffers: list[torch.Tensor] = []
        self._materialized_buffers: list[torch.Tensor] = []
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    def transform_model(self, model: nn.Module) -> nn.Module:
        assert self.runtime is not None
        self.dp_group = self.runtime.get_group(MeshAxis.DP)
        if self.dp_group is None:
            raise ValueError("Zero3Plugin requires mesh.dp > 1")
        self.world_size = dist.get_world_size(self.dp_group)
        self.rank = dist.get_rank(self.dp_group)
        self._prepare_buckets(model)
        self.optimizer = self.runtime.create_optimizer([bucket.local_param for bucket in self.buckets])
        self.scheduler = self.runtime.create_scheduler(self.optimizer)
        return model

    def on_phase(self, phase: RuntimePhase) -> None:
        if phase == RuntimePhase.PRE_FORWARD:
            assert self.runtime is not None
            context = self.runtime.state.step_context
            self._reset_buckets(
                grad_accum_start=context.accum_start if context is not None else True,
                grad_accum_end=context.is_step_boundary if context is not None else True,
            )
            if self.enable_prefetch and self.buckets:
                self._prefetch_bucket(self.buckets[0], direction="forward")
        elif phase == RuntimePhase.POST_BACKWARD:
            assert self.runtime is not None
            if self.runtime.state.step_context.is_step_boundary:
                self._wait_all_grad_sync()

    def _prepare_buckets(self, model: nn.Module) -> None:
        visited: set[str] = set()
        param_to_name = {id(param): name for name, param in model.named_parameters()}
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

        if not bucket_specs:
            return

        dtype, device = bucket_specs[0][1][0].dtype, bucket_specs[0][1][0].device
        max_buffer_size = max(self._padded_len(sum(param.numel() for param in params)) for _, params, _ in bucket_specs)
        self.data_buffers = [
            torch.empty(max_buffer_size, dtype=dtype, device=device),
            torch.empty(max_buffer_size, dtype=dtype, device=device),
        ]

        for index, (module, params, logical_names) in enumerate(bucket_specs):
            bucket = self._make_bucket(index, module, params, logical_names)
            self.buckets.append(bucket)

        for index, bucket in enumerate(self.buckets):
            if index > 0:
                bucket.prev_bucket = self.buckets[index - 1]
            if index + 1 < len(self.buckets):
                bucket.next_bucket = self.buckets[index + 1]

        self._reset_buckets(grad_accum_start=True, grad_accum_end=True)
        self._add_hooks()
        for bucket in self.buckets:
            self._free_full_params(bucket)

    def _make_bucket(self, index: int, module: nn.Module, params: list[nn.Parameter], logical_names: list[str]) -> _Bucket:
        dtype, device = params[0].dtype, params[0].device
        param_shapes = [param.shape for param in params]
        param_numels = [param.numel() for param in params]
        buffer_size = self._padded_len(sum(param_numels))
        full_param = torch.zeros(buffer_size, dtype=dtype, device=device)
        offset = 0
        for param in params:
            full_param[offset : offset + param.numel()].copy_(param.detach().view(-1))
            offset += param.numel()

        shard_len = buffer_size // self.world_size
        shard_start = self.rank * shard_len
        shard_end = (self.rank + 1) * shard_len
        local_param = nn.Parameter(full_param[shard_start:shard_end].clone())
        data_buffer = self.data_buffers[index % len(self.data_buffers)][:buffer_size]
        grad_buffer = torch.empty(buffer_size, dtype=dtype, device=device)
        return _Bucket(
            module=module,
            params=params,
            param_shapes=param_shapes,
            param_numels=param_numels,
            buffer_size=buffer_size,
            local_param=local_param,
            data_buffer=data_buffer,
            grad_buffer=grad_buffer,
            index=index,
            logical_names=logical_names,
            pending=len(params),
        )

    def _add_hooks(self) -> None:
        for bucket in self.buckets:
            bucket.module.register_forward_pre_hook(self._make_materialize_forward_hook(bucket))
            bucket.module.register_forward_hook(self._make_free_forward_hook(bucket))
            bucket.module.register_full_backward_pre_hook(self._make_materialize_backward_hook(bucket))
            bucket.module.register_full_backward_hook(self._make_free_backward_hook(bucket))
            for param in bucket.params:
                param.register_hook(self._make_attach_grad_hook(bucket))
                param.register_post_accumulate_grad_hook(self._make_reduce_grad_hook(bucket))

    def _make_materialize_forward_hook(self, bucket: _Bucket):
        def hook(_module: nn.Module, _inputs) -> None:
            self._materialize_full_params(bucket, direction="forward")
            if self.enable_prefetch and bucket.next_bucket is not None:
                self._prefetch_bucket(bucket.next_bucket, direction="forward")

        return hook

    def _make_free_forward_hook(self, bucket: _Bucket):
        def hook(_module: nn.Module, _inputs, _outputs) -> None:
            self._free_full_params(bucket)
            bucket.fwd_handle = None

        return hook

    def _make_materialize_backward_hook(self, bucket: _Bucket):
        def hook(_module: nn.Module, _grad_outputs) -> None:
            self._materialize_full_params(bucket, direction="backward")
            if self.enable_prefetch and bucket.prev_bucket is not None:
                self._prefetch_bucket(bucket.prev_bucket, direction="backward")

        return hook

    def _make_free_backward_hook(self, bucket: _Bucket):
        def hook(_module: nn.Module, _grad_inputs, _grad_outputs) -> None:
            pass

        return hook

    def _make_attach_grad_hook(self, bucket: _Bucket):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if not bucket.attached:
                offset = 0
                for param, numel, shape in zip(bucket.params, bucket.param_numels, bucket.param_shapes):
                    param.grad = bucket.grad_buffer[offset : offset + numel].view(shape)
                    offset += numel
                bucket.attached = True
            return grad

        return hook

    def _make_reduce_grad_hook(self, bucket: _Bucket):
        def hook(_param: nn.Parameter) -> None:
            bucket.pending -= 1
            if bucket.pending == 0:
                if bucket.local_param.grad is None:
                    bucket.local_param.grad = torch.empty_like(bucket.local_param.data)
                bucket.grad_handle = self._reduce_scatter_avg(bucket)

        return hook

    def _prefetch_bucket(self, bucket: _Bucket, direction: str) -> None:
        assert self.dp_group is not None
        if direction == "forward":
            if bucket.fwd_handle is not None:
                return
            bucket.fwd_handle = dist.all_gather_into_tensor(
                bucket.data_buffer,
                bucket.local_param.detach().contiguous(),
                group=self.dp_group,
                async_op=True,
            )
            return
        if direction == "backward":
            if bucket.bwd_handle is not None:
                return
            bucket.bwd_handle = dist.all_gather_into_tensor(
                bucket.data_buffer,
                bucket.local_param.detach().contiguous(),
                group=self.dp_group,
                async_op=True,
            )
            return
        raise ValueError(f"unknown ZeRO3 prefetch direction={direction}")

    def _materialize_full_params(self, bucket: _Bucket, direction: str) -> None:
        if direction == "forward":
            if bucket.fwd_handle is None:
                self._prefetch_bucket(bucket, direction="forward")
            assert bucket.fwd_handle is not None
            bucket.fwd_handle.wait()
        elif direction == "backward":
            if bucket.bwd_handle is None:
                self._prefetch_bucket(bucket, direction="backward")
            assert bucket.bwd_handle is not None
            bucket.bwd_handle.wait()
        else:
            raise ValueError(f"unknown ZeRO3 materialize direction={direction}")
        self._bind_full_params(bucket, bucket.data_buffer)

    def _materialize_full_params_sync(self, bucket: _Bucket) -> None:
        assert self.dp_group is not None
        buffer = torch.empty(bucket.buffer_size, dtype=bucket.local_param.dtype, device=bucket.local_param.device)
        dist.all_gather_into_tensor(
            buffer,
            bucket.local_param.detach().contiguous(),
            group=self.dp_group,
        )
        self._materialized_buffers.append(buffer)
        self._bind_full_params(bucket, buffer)

    def _bind_full_params(self, bucket: _Bucket, buffer: torch.Tensor) -> None:
        offset = 0
        for param, numel, shape in zip(bucket.params, bucket.param_numels, bucket.param_shapes):
            param.data = buffer[offset : offset + numel].view(shape)
            offset += numel

    def _free_full_params(self, bucket: _Bucket) -> None:
        for param in bucket.params:
            param.data = bucket.local_param.data

    def _reduce_scatter_avg(self, bucket: _Bucket) -> dist.Work | _AllReduceShardWork:
        assert self.dp_group is not None
        if dist.get_backend(self.dp_group) == "gloo":
            work = dist.all_reduce(bucket.grad_buffer, op=dist.ReduceOp.AVG, group=self.dp_group, async_op=True)
            shard_len = bucket.local_param.numel()
            shard_start = self.rank * shard_len
            shard_end = (self.rank + 1) * shard_len
            return _AllReduceShardWork(work, bucket.grad_buffer, bucket.local_param.grad, shard_start, shard_end)
        return dist.reduce_scatter_tensor(
            bucket.local_param.grad,
            bucket.grad_buffer,
            op=dist.ReduceOp.AVG,
            group=self.dp_group,
            async_op=True,
        )

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
                "rank": self.rank,
                "world_size": self.world_size,
                "shard_offset": self.rank * shard_len,
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

    def _wait_all_grad_sync(self) -> None:
        for bucket in self.buckets:
            if bucket.grad_handle is None:
                raise RuntimeError("ZeRO3 bucket grad handle is None after backward")
            bucket.grad_handle.wait()
            self._free_full_params(bucket)

    def _reset_buckets(self, *, grad_accum_start: bool, grad_accum_end: bool) -> None:
        self._materialized_buffers.clear()
        for bucket in self.buckets:
            if grad_accum_start:
                bucket.grad_buffer.zero_()
            bucket.pending = len(bucket.params) if grad_accum_end else 0
            bucket.attached = False
            bucket.grad_handle = None
            bucket.fwd_handle = None
            bucket.bwd_handle = None

    def _padded_len(self, numel: int) -> int:
        return (numel + self.world_size - 1) // self.world_size * self.world_size
