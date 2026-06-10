from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MethodType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime.core import RuntimeCore

import torch
import torch.distributed as dist
import torch.nn as nn

from runtime.buffer_allocator import allocate_buffer
from runtime.core import PpStatus
from parallel.specs import TpSpParallelSpec
from runtime.mesh import MeshAxis
from runtime.plugin import PipelineParallelizableModule, PluginId, RuntimePlugin, TpSpParallelizableModule


@dataclass
class _PipelineMicroState:
    input_activation: torch.Tensor | None = None
    output_activation: torch.Tensor | None = None
    raw_loss: torch.Tensor | None = None
    activation_send_buffer: torch.Tensor | None = None
    activation_send_work: dist.Work | None = None
    grad_send_buffer: torch.Tensor | None = None
    grad_send_work: dist.Work | None = None


class _PipelineScheduleKind(str, Enum):
    AFAB = "afab"
    ONE_FWD_ONE_BWD = "1f1b"


class _PipelineActionKind(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass(frozen=True)
class _PipelineAction:
    kind: _PipelineActionKind
    microbatch_idx: int
    backward_step_idx: int = 0


class PipelineParallelPlugin(RuntimePlugin):
    def __init__(self, schedule: str = "afab") -> None:
        super().__init__(
            id=PluginId.PP,
            name="pipeline_parallel",
            runs_before={
                PluginId.DP,
                PluginId.ZERO1,
                PluginId.ZERO2,
                PluginId.ZERO3,
                PluginId.TP,
                PluginId.SP,
                PluginId.EP,
                PluginId.PRECISION,
            },
        )
        self.stage_index = 0
        self.stage_count = 1
        self.prev_global_rank: int | None = None
        self.next_global_rank: int | None = None
        self.last_global_rank = 0
        self.hidden_size = 0
        self.pp_group: dist.ProcessGroup | None = None
        self.sequence_parallel_enabled = False
        self.sequence_parallel_world_size = 1
        self.schedule = _PipelineScheduleKind(schedule)

    def bind(self, runtime: "RuntimeCore") -> None:
        super().bind(runtime)
        active_plugins = {plugin.id for plugin in runtime.plugins if plugin is not self}
        self.sequence_parallel_enabled = PluginId.SP in active_plugins
        self.sequence_parallel_world_size = runtime.mesh.tp if self.sequence_parallel_enabled else 1

    def transform_model(self, model: nn.Module) -> nn.Module:
        assert self.runtime is not None
        self._validate_runtime_support()
        if not dist.is_initialized():
            raise ValueError("PipelineParallelPlugin requires torch.distributed to be initialized")

        self.pp_group = self.runtime.get_group(MeshAxis.PP)
        global_rank = dist.get_rank()
        _, pp_idx, _, _ = self.runtime.mesh.rank_coordinates(global_rank)
        self.stage_index = pp_idx
        self.stage_count = self.runtime.mesh.pp
        self.prev_global_rank = self._stage_global_rank(pp_idx - 1) if pp_idx > 0 else None
        self.next_global_rank = self._stage_global_rank(pp_idx + 1) if pp_idx + 1 < self.stage_count else None
        self.last_global_rank = self._stage_global_rank(self.stage_count - 1)
        if not isinstance(model, PipelineParallelizableModule):
            raise TypeError(
                "PipelineParallelPlugin requires model.pipeline_parallel_spec(), "
                f"got {type(model).__name__}"
            )
        spec = model.pipeline_parallel_spec()
        self.hidden_size = _infer_hidden_size(model, spec)
        self._partition_model(model, spec, pp_idx, self.stage_count)
        self._filter_tpsp_spec(model, spec, pp_idx, self.stage_count)
        return model

    def build_step_runner(self):
        return self._run_pipeline_step

    def _run_pipeline_step(self, batch) -> torch.Tensor:
        assert self.runtime is not None
        context = self.runtime.state.step_context
        num_microbatches = self.runtime.plan.pp_schedule.microbatches
        micro_batches = _split_batch(batch, num_microbatches)
        if len(micro_batches) != num_microbatches:
            raise ValueError(
                "PipelineParallelPlugin microbatch split count does not match plan.pp_schedule.microbatches: "
                f"{len(micro_batches)} vs {num_microbatches}"
            )
        states = [_PipelineMicroState() for _ in range(num_microbatches)]
        total_loss: torch.Tensor | None = None

        for action in self._build_schedule(num_microbatches):
            if action.kind == _PipelineActionKind.FORWARD:
                total_loss = self._run_forward_action(
                    micro_batches=micro_batches,
                    states=states,
                    action=action,
                    total_loss=total_loss,
                )
            else:
                self._run_backward_action(states=states, action=action)

        for state in states:
            if state.activation_send_work is not None:
                state.activation_send_work.wait()
            if state.grad_send_work is not None:
                state.grad_send_work.wait()

        loss_out = self._broadcast_loss(total_loss, micro_batches[0])
        self.runtime.state.loss = loss_out
        return loss_out

    def _build_schedule(self, num_microbatches: int) -> list[_PipelineAction]:
        if self.schedule == _PipelineScheduleKind.AFAB:
            return self._build_afab_schedule(num_microbatches)
        if self.schedule == _PipelineScheduleKind.ONE_FWD_ONE_BWD:
            return self._build_1f1b_schedule(num_microbatches)
        raise ValueError(f"unsupported pipeline schedule={self.schedule.value}")

    def _build_afab_schedule(self, num_microbatches: int) -> list[_PipelineAction]:
        actions = [
            _PipelineAction(kind=_PipelineActionKind.FORWARD, microbatch_idx=micro_idx)
            for micro_idx in range(num_microbatches)
        ]
        actions.extend(
            _PipelineAction(
                kind=_PipelineActionKind.BACKWARD,
                microbatch_idx=micro_idx,
                backward_step_idx=backward_step_idx,
            )
            for backward_step_idx, micro_idx in enumerate(range(num_microbatches - 1, -1, -1))
        )
        return actions

    def _build_1f1b_schedule(self, num_microbatches: int) -> list[_PipelineAction]:
        warmup = min(self.stage_count - self.stage_index - 1, num_microbatches)
        remaining = num_microbatches - warmup
        actions: list[_PipelineAction] = []
        for micro_idx in range(warmup):
            actions.append(_PipelineAction(kind=_PipelineActionKind.FORWARD, microbatch_idx=micro_idx))
        for backward_step_idx in range(remaining):
            forward_microbatch_idx = warmup + backward_step_idx
            if forward_microbatch_idx < num_microbatches:
                actions.append(
                    _PipelineAction(
                        kind=_PipelineActionKind.FORWARD,
                        microbatch_idx=forward_microbatch_idx,
                    )
                )
            actions.append(
                _PipelineAction(
                    kind=_PipelineActionKind.BACKWARD,
                    microbatch_idx=backward_step_idx,
                    backward_step_idx=backward_step_idx,
                )
            )
        for backward_step_idx in range(remaining, num_microbatches):
            actions.append(
                _PipelineAction(
                    kind=_PipelineActionKind.BACKWARD,
                    microbatch_idx=backward_step_idx,
                    backward_step_idx=backward_step_idx,
                )
            )
        return actions

    def _run_forward_action(
        self,
        *,
        micro_batches,
        states: list[_PipelineMicroState],
        action: _PipelineAction,
        total_loss: torch.Tensor | None,
    ) -> torch.Tensor | None:
        assert self.runtime is not None
        context = self.runtime.state.step_context
        context.set_pp_state(microbatch_idx=action.microbatch_idx, status=PpStatus.FORWARD)
        micro_batch = micro_batches[action.microbatch_idx]
        state = states[action.microbatch_idx]

        if self.prev_global_rank is None:
            model_input = micro_batch
            input_activation = None
        else:
            input_activation, recv_work = self._recv_activation_async(micro_batch)
            recv_work.wait()
            input_activation.requires_grad_(True)
            model_input = {"hidden_states": input_activation}
            if isinstance(micro_batch, dict) and "position_offset" in micro_batch:
                model_input["position_offset"] = micro_batch["position_offset"]
            if isinstance(micro_batch, dict) and "position_ids" in micro_batch:
                model_input["position_ids"] = micro_batch["position_ids"]
            if self.next_global_rank is None:
                model_input["labels"] = _extract_labels(micro_batch)
                if isinstance(micro_batch, dict) and "loss_weight" in micro_batch:
                    model_input["loss_weight"] = micro_batch["loss_weight"]

        self.runtime._forward_step_impl(model_input)
        state.input_activation = input_activation

        if self.next_global_rank is None:
            if not torch.is_tensor(self.runtime.state.loss):
                raise TypeError("last PP stage must return a Tensor loss")
            state.raw_loss = self.runtime.state.loss
            return self.runtime.state.loss.detach() if total_loss is None else total_loss + self.runtime.state.loss.detach()

        if not torch.is_tensor(self.runtime.state.outputs):
            raise TypeError("non-last PP stage must return Tensor activations")
        boundary_activation = _cast_boundary_activation(self.runtime.state.outputs, self.runtime)
        self.runtime.state.outputs = boundary_activation
        state.output_activation = boundary_activation
        send_buffer, send_work = self._send_activation_async(boundary_activation.detach())
        state.activation_send_buffer = send_buffer
        state.activation_send_work = send_work
        return total_loss

    def _run_backward_action(
        self,
        *,
        states: list[_PipelineMicroState],
        action: _PipelineAction,
    ) -> None:
        assert self.runtime is not None
        context = self.runtime.state.step_context
        num_microbatches = self.runtime.plan.pp_schedule.microbatches
        if action.backward_step_idx == 0:
            status = PpStatus.BACKWARD_START
        elif action.backward_step_idx == num_microbatches - 1:
            status = PpStatus.BACKWARD_END
        else:
            status = PpStatus.BACKWARD_MIDDLE
        context.set_pp_state(
            microbatch_idx=action.microbatch_idx,
            status=status,
        )
        state = states[action.microbatch_idx]
        if self.next_global_rank is None:
            assert state.raw_loss is not None
            self.runtime.state.loss = state.raw_loss / float(num_microbatches)
            self.runtime._backward_step_impl()
            if self.prev_global_rank is not None:
                assert state.input_activation is not None and state.input_activation.grad is not None
                send_buffer, send_work = self._send_grad_async(state.input_activation.grad.detach())
                state.grad_send_buffer = send_buffer
                state.grad_send_work = send_work
            return

        assert state.output_activation is not None
        self.runtime.state.outputs = state.output_activation
        grad_output, recv_work = self._recv_grad_async(state.output_activation)
        recv_work.wait()
        self.runtime._backward_step_impl(grad_output=grad_output)
        if self.prev_global_rank is not None:
            assert state.input_activation is not None and state.input_activation.grad is not None
            send_buffer, send_work = self._send_grad_async(state.input_activation.grad.detach())
            state.grad_send_buffer = send_buffer
            state.grad_send_work = send_work

    def _broadcast_loss(self, total_loss: torch.Tensor | None, batch) -> torch.Tensor:
        assert self.runtime is not None
        device = _model_device(self.runtime.model)
        dtype = torch.float32 if total_loss is None else total_loss.dtype
        if self.next_global_rank is None:
            assert total_loss is not None
            loss = total_loss / float(self.runtime.plan.pp_schedule.microbatches)
        else:
            loss = torch.zeros((), device=device, dtype=dtype)
        if self.pp_group is not None and self.stage_count > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.pp_group)
        return loss

    def _recv_activation_async(self, batch) -> tuple[torch.Tensor, dist.Work]:
        assert self.prev_global_rank is not None
        assert self.runtime is not None
        microbatch_idx = self.runtime.state.step_context.pp_cur_microbatch_idx
        device = _model_device(self.runtime.model)
        dtype = _activation_dtype(self.runtime)
        shape = _activation_shape(
            batch,
            self.hidden_size,
            sequence_parallel_world_size=self.sequence_parallel_world_size,
        )
        buffer = allocate_buffer(
            key=f"pp.recv_activation.mb{microbatch_idx}",
            shape=shape,
            dtype=dtype,
            device=device,
        )
        work = dist.irecv(buffer, src=self.prev_global_rank)
        return buffer, work

    def _send_activation_async(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dist.Work]:
        assert self.next_global_rank is not None
        buffer = tensor.contiguous()
        work = dist.isend(buffer, dst=self.next_global_rank)
        return buffer, work

    def _recv_grad_async(self, output_activation: torch.Tensor) -> tuple[torch.Tensor, dist.Work]:
        assert self.next_global_rank is not None
        assert self.runtime is not None
        microbatch_idx = self.runtime.state.step_context.pp_cur_microbatch_idx
        grad = allocate_buffer(
            key=f"pp.recv_grad.mb{microbatch_idx}",
            shape=tuple(output_activation.shape),
            dtype=output_activation.dtype,
            device=output_activation.device,
        )
        work = dist.irecv(grad, src=self.next_global_rank)
        return grad, work

    def _send_grad_async(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dist.Work]:
        assert self.prev_global_rank is not None
        buffer = tensor.contiguous()
        work = dist.isend(buffer, dst=self.prev_global_rank)
        return buffer, work

    def _stage_global_rank(self, stage_index: int) -> int:
        assert self.runtime is not None
        global_rank = dist.get_rank()
        dp_idx, _, cp_idx, tp_idx = self.runtime.mesh.rank_coordinates(global_rank)
        return self.runtime.mesh.rank_id(dp=dp_idx, pp=stage_index, cp=cp_idx, tp=tp_idx)

    def _validate_runtime_support(self) -> None:
        assert self.runtime is not None
        mesh = self.runtime.mesh
        if mesh.pp <= 1:
            raise ValueError("PipelineParallelPlugin requires mesh.pp > 1")
        unsupported = set()
        active = {plugin.id for plugin in self.runtime.plugins if plugin is not self}
        overlap = sorted(plugin_id.value for plugin_id in active & unsupported)
        if overlap:
            raise ValueError(f"PipelineParallelPlugin does not yet support plugin combinations: {overlap}")

    def _partition_model(self, model: nn.Module, spec, stage_index: int, stage_count: int) -> None:
        assert self.runtime is not None
        for path in spec.head_layers:
            if stage_index != 0:
                self.runtime.mark_module_path_omitted(path)
                _replace_module_path(model, path, None)
        for path in spec.tail_layers:
            if stage_index != stage_count - 1:
                self.runtime.mark_module_path_omitted(path)
                _replace_module_path(model, path, None)

        for path in spec.pipe_layers:
            module = model.get_submodule(path)
            if not isinstance(module, nn.ModuleList):
                raise TypeError(
                    f"PipelineParallelPlugin expects pipe layer path={path!r} to be nn.ModuleList, "
                    f"got {type(module).__name__}"
                )
            start, end = _layer_range(len(module), stage_index, stage_count)
            for layer_idx in range(len(module)):
                if not start <= layer_idx < end:
                    self.runtime.mark_module_path_omitted(f"{path}.{layer_idx}")
            partitioned = nn.ModuleList(
                [layer if start <= layer_idx < end else _IdentityPipeLayer() for layer_idx, layer in enumerate(module)]
            )
            _replace_module_path(model, path, partitioned)

    def _filter_tpsp_spec(self, model: nn.Module, pp_spec, stage_index: int, stage_count: int) -> None:
        if not isinstance(model, TpSpParallelizableModule):
            return
        tpsp_spec = model.tpsp_parallelize_spec()
        keep_prefixes: list[str] = []
        if stage_index == 0:
            keep_prefixes.extend(pp_spec.head_layers)
        if stage_index == stage_count - 1:
            keep_prefixes.extend(pp_spec.tail_layers)
        for path in pp_spec.pipe_layers:
            module = model.get_submodule(path)
            if not isinstance(module, nn.ModuleList):
                continue
            start, end = _layer_range(len(module), stage_index, stage_count)
            keep_prefixes.extend(f"{path}.{layer_idx}" for layer_idx in range(start, end))

        filtered_rules = [rule for rule in tpsp_spec.rules if _path_matches_any_prefix(rule.module_path, keep_prefixes)]
        filtered_tie_rules = [
            tie_rule
            for tie_rule in tpsp_spec.tie_rules
            if _path_matches_any_prefix(tie_rule[0], keep_prefixes)
            and _path_matches_any_prefix(tie_rule[1], keep_prefixes)
        ]
        filtered_spec = TpSpParallelSpec(rules=filtered_rules, tie_rules=filtered_tie_rules)
        model.tpsp_parallelize_spec = MethodType(lambda _self: filtered_spec, model)


def _layer_range(num_layers: int, stage_index: int, stage_count: int) -> tuple[int, int]:
    base, remainder = divmod(num_layers, stage_count)
    start = stage_index * base + min(stage_index, remainder)
    width = base + (1 if stage_index < remainder else 0)
    return start, start + width


def _path_matches_any_prefix(path: str, prefixes: list[str]) -> bool:
    for prefix in prefixes:
        if path == prefix or path.startswith(prefix + "."):
            return True
    return False


class _IdentityPipeLayer(nn.Module):
    def forward(self, *args, **kwargs):
        if not args:
            raise ValueError("identity pipe layer expects at least one positional argument")
        return args[0]


def _split_batch(batch, num_microbatches: int):
    if num_microbatches == 1:
        return [batch]
    batch_size = _batch_size(batch)
    if batch_size % num_microbatches != 0:
        raise ValueError(
            f"pipeline microbatch split requires batch size divisible by microbatches, "
            f"got batch_size={batch_size}, microbatches={num_microbatches}"
        )
    micro_batch_size = batch_size // num_microbatches
    return [_slice_batch(batch, index * micro_batch_size, micro_batch_size) for index in range(num_microbatches)]


def _batch_size(batch) -> int:
    if isinstance(batch, dict):
        input_ids = batch.get("input_ids")
        if torch.is_tensor(input_ids):
            return int(input_ids.size(0))
    if isinstance(batch, (tuple, list)) and batch and torch.is_tensor(batch[0]):
        return int(batch[0].size(0))
    if torch.is_tensor(batch):
        return int(batch.size(0))
    raise TypeError(f"unsupported PP batch type={type(batch).__name__}")


def _slice_batch(batch, start: int, length: int):
    if isinstance(batch, dict):
        return {
            key: value.narrow(0, start, length).contiguous() if torch.is_tensor(value) and value.size(0) >= start + length else value
            for key, value in batch.items()
        }
    if isinstance(batch, tuple):
        return tuple(_slice_batch_item(value, start, length) for value in batch)
    if isinstance(batch, list):
        return [_slice_batch_item(value, start, length) for value in batch]
    if torch.is_tensor(batch):
        return batch.narrow(0, start, length).contiguous()
    raise TypeError(f"unsupported PP batch type={type(batch).__name__}")


def _slice_batch_item(value, start: int, length: int):
    if torch.is_tensor(value):
        return value.narrow(0, start, length).contiguous()
    return value


def _unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, dict):
        return batch["input_ids"], batch.get("labels")
    if isinstance(batch, (tuple, list)):
        if len(batch) != 2:
            raise ValueError(f"expected (input_ids, labels) batch tuple, got len={len(batch)}")
        return batch[0], batch[1]
    if torch.is_tensor(batch):
        return batch, None
    raise TypeError(f"unsupported PP batch type={type(batch).__name__}")


def _extract_labels(batch) -> torch.Tensor | None:
    _input_ids, labels = _unpack_batch(batch)
    return labels


def _activation_shape(
    batch,
    hidden_size: int,
    *,
    sequence_parallel_world_size: int = 1,
) -> tuple[int, int, int]:
    input_ids, _labels = _unpack_batch(batch)
    seq_len = int(input_ids.size(1))
    if sequence_parallel_world_size > 1:
        if seq_len % sequence_parallel_world_size != 0:
            raise ValueError(
                "PP+SP activation shape requires sequence length divisible by tp world size, "
                f"got seq_len={seq_len}, tp={sequence_parallel_world_size}"
            )
        seq_len //= sequence_parallel_world_size
    return int(input_ids.size(0)), seq_len, hidden_size


def _model_device(model: nn.Module) -> torch.device:
    first_param = next(model.parameters(), None)
    if first_param is None:
        raise ValueError("PP model must have parameters")
    return first_param.device


def _infer_hidden_size(model: nn.Module, spec) -> int:
    for path in spec.head_layers:
        module = model.get_submodule(path)
        if isinstance(module, nn.Embedding):
            return int(module.embedding_dim)
    for path in spec.tail_layers:
        module = model.get_submodule(path)
        weight = getattr(module, "weight", None)
        if torch.is_tensor(weight) and weight.ndim >= 1:
            return int(weight.shape[0])
    raise ValueError("unable to infer PP hidden size from model/spec")


def _replace_module_path(model: nn.Module, path: str, module: nn.Module | None) -> None:
    if "." not in path:
        setattr(model, path, module)
        return
    parent_path, leaf = path.rsplit(".", 1)
    parent = model.get_submodule(parent_path)
    setattr(parent, leaf, module)


def _activation_dtype(runtime) -> torch.dtype:
    for plugin in runtime.plugins:
        compute_dtype = getattr(plugin, "compute_dtype", None)
        if compute_dtype is not None:
            return compute_dtype
    first_param = next(runtime.model.parameters(), None)
    if first_param is None:
        raise ValueError("PP model must have parameters")
    return first_param.dtype


def _cast_boundary_activation(tensor: torch.Tensor, runtime) -> torch.Tensor:
    target_dtype = _activation_dtype(runtime)
    if tensor.dtype == target_dtype:
        return tensor
    return tensor.to(dtype=target_dtype)
