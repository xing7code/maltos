from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from utils.constants import (
    HIDDEN_STATES_KEY,
    INPUT_IDS_KEY,
    LABELS_KEY,
    LOSS_WEIGHT_KEY,
    POSITION_IDS_KEY,
    POSITION_OFFSET_KEY,
    SEQUENCE_IDS_KEY,
)

from runtime.buffer_allocator import BufferPolicy, acquire_buffer
from runtime.step_runners.base import DefaultStepRunner
from runtime.types import LossOutput, PipelineOutput, PpStatus
from utils.distributed import all_reduce_tensor, irecv_tensor_async, isend_tensor_async

if TYPE_CHECKING:
    from runtime.plugins.pp import PipelineParallelPlugin


@dataclass
class PipelineMicroState:
    input_activation: torch.Tensor | None = None
    output_activation: torch.Tensor | None = None
    raw_loss: torch.Tensor | None = None
    auxiliary_loss: torch.Tensor | None = None
    activation_send_buffer: torch.Tensor | None = None
    activation_send_work: object | None = None
    grad_send_buffer: torch.Tensor | None = None
    grad_send_work: object | None = None


class PipelineScheduleKind(str, Enum):
    AFAB = "afab"
    ONE_FWD_ONE_BWD = "1f1b"


class PipelineActionKind(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass(frozen=True)
class PipelineAction:
    kind: PipelineActionKind
    microbatch_idx: int
    backward_step_idx: int = 0


@dataclass(frozen=True)
class PipelineStepRunner:
    plugin: "PipelineParallelPlugin"

    def run(self, runtime, batch) -> torch.Tensor:
        context = runtime.state.step_context
        num_microbatches = runtime.plan.pp_schedule.microbatches
        micro_batches = split_batch(batch, num_microbatches)
        if len(micro_batches) != num_microbatches:
            raise ValueError(
                "PipelineStepRunner microbatch split count does not match plan.pp_schedule.microbatches: "
                f"{len(micro_batches)} vs {num_microbatches}"
            )
        states = [PipelineMicroState() for _ in range(num_microbatches)]
        total_loss: torch.Tensor | None = None
        total_auxiliary_loss: torch.Tensor | None = None

        for action in self.build_schedule(num_microbatches):
            if action.kind == PipelineActionKind.FORWARD:
                total_loss, total_auxiliary_loss = self.run_forward_action(
                    runtime=runtime,
                    micro_batches=micro_batches,
                    states=states,
                    action=action,
                    total_loss=total_loss,
                    total_auxiliary_loss=total_auxiliary_loss,
                )
            else:
                self.run_backward_action(
                    runtime=runtime,
                    states=states,
                    action=action,
                )

        for state in states:
            if state.activation_send_work is not None:
                state.activation_send_work.wait()
            if state.grad_send_work is not None:
                state.grad_send_work.wait()

        loss_out = self.broadcast_loss(
            runtime=runtime,
            total_loss=total_loss,
            total_auxiliary_loss=total_auxiliary_loss,
            batch=micro_batches[0],
        )
        runtime.state.loss = loss_out
        return loss_out

    def build_schedule(self, num_microbatches: int) -> list[PipelineAction]:
        if self.plugin.schedule == PipelineScheduleKind.AFAB:
            return self.build_afab_schedule(num_microbatches)
        if self.plugin.schedule == PipelineScheduleKind.ONE_FWD_ONE_BWD:
            return self.build_1f1b_schedule(num_microbatches)
        raise ValueError(f"unsupported pipeline schedule={self.plugin.schedule.value}")

    @staticmethod
    def build_afab_schedule(num_microbatches: int) -> list[PipelineAction]:
        actions = [
            PipelineAction(kind=PipelineActionKind.FORWARD, microbatch_idx=micro_idx)
            for micro_idx in range(num_microbatches)
        ]
        actions.extend(
            PipelineAction(
                kind=PipelineActionKind.BACKWARD,
                microbatch_idx=micro_idx,
                backward_step_idx=backward_step_idx,
            )
            for backward_step_idx, micro_idx in enumerate(range(num_microbatches - 1, -1, -1))
        )
        return actions

    def build_1f1b_schedule(self, num_microbatches: int) -> list[PipelineAction]:
        warmup = min(self.plugin.stage_count - self.plugin.stage_index - 1, num_microbatches)
        remaining = num_microbatches - warmup
        actions: list[PipelineAction] = []
        for micro_idx in range(warmup):
            actions.append(PipelineAction(kind=PipelineActionKind.FORWARD, microbatch_idx=micro_idx))
        for backward_step_idx in range(remaining):
            forward_microbatch_idx = warmup + backward_step_idx
            if forward_microbatch_idx < num_microbatches:
                actions.append(
                    PipelineAction(
                        kind=PipelineActionKind.FORWARD,
                        microbatch_idx=forward_microbatch_idx,
                    )
                )
            actions.append(
                PipelineAction(
                    kind=PipelineActionKind.BACKWARD,
                    microbatch_idx=backward_step_idx,
                    backward_step_idx=backward_step_idx,
                )
            )
        for backward_step_idx in range(remaining, num_microbatches):
            actions.append(
                PipelineAction(
                    kind=PipelineActionKind.BACKWARD,
                    microbatch_idx=backward_step_idx,
                    backward_step_idx=backward_step_idx,
                )
            )
        return actions

    def run_forward_action(
        self,
        *,
        runtime,
        micro_batches,
        states: list[PipelineMicroState],
        action: PipelineAction,
        total_loss: torch.Tensor | None,
        total_auxiliary_loss: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        context = runtime.state.step_context
        context.set_pp_state(microbatch_idx=action.microbatch_idx, status=PpStatus.FORWARD)
        micro_batch = micro_batches[action.microbatch_idx]
        state = states[action.microbatch_idx]

        if self.plugin.prev_global_rank is None:
            model_input = micro_batch
            input_activation = None
        else:
            input_activation, recv_work = self.recv_activation_async(runtime, micro_batch)
            recv_work.wait()
            input_activation.requires_grad_(True)
            model_input = {HIDDEN_STATES_KEY: input_activation}
            if isinstance(micro_batch, dict) and POSITION_OFFSET_KEY in micro_batch:
                model_input[POSITION_OFFSET_KEY] = micro_batch[POSITION_OFFSET_KEY]
            if isinstance(micro_batch, dict) and POSITION_IDS_KEY in micro_batch:
                model_input[POSITION_IDS_KEY] = micro_batch[POSITION_IDS_KEY]
            if isinstance(micro_batch, dict) and SEQUENCE_IDS_KEY in micro_batch:
                model_input[SEQUENCE_IDS_KEY] = micro_batch[SEQUENCE_IDS_KEY]
            if self.plugin.next_global_rank is None:
                model_input[LABELS_KEY] = extract_labels(micro_batch)
                if isinstance(micro_batch, dict) and LOSS_WEIGHT_KEY in micro_batch:
                    model_input[LOSS_WEIGHT_KEY] = micro_batch[LOSS_WEIGHT_KEY]

        DefaultStepRunner.run_forward(runtime, model_input)
        state.input_activation = input_activation

        if self.plugin.next_global_rank is None:
            if not torch.is_tensor(runtime.state.loss):
                raise TypeError("last PP stage must return a Tensor loss")
            state.raw_loss = runtime.state.loss
            if isinstance(runtime.state.outputs, LossOutput) and runtime.state.outputs.auxiliary_loss is not None:
                state.auxiliary_loss = runtime.state.outputs.auxiliary_loss
                detached_auxiliary_loss = state.auxiliary_loss.detach()
                total_auxiliary_loss = (
                    detached_auxiliary_loss
                    if total_auxiliary_loss is None
                    else total_auxiliary_loss + detached_auxiliary_loss
                )
            return (
                runtime.state.loss.detach() if total_loss is None else total_loss + runtime.state.loss.detach(),
                total_auxiliary_loss,
            )

        if isinstance(runtime.state.outputs, PipelineOutput):
            boundary_activation = cast_boundary_activation(runtime.state.outputs.activation, runtime)
            state.auxiliary_loss = runtime.state.outputs.auxiliary_loss
            detached_auxiliary_loss = state.auxiliary_loss.detach()
            total_auxiliary_loss = (
                detached_auxiliary_loss
                if total_auxiliary_loss is None
                else total_auxiliary_loss + detached_auxiliary_loss
            )
        elif torch.is_tensor(runtime.state.outputs):
            boundary_activation = cast_boundary_activation(runtime.state.outputs, runtime)
        else:
            raise TypeError("non-last PP stage must return Tensor or PipelineOutput activations")
        runtime.state.outputs = boundary_activation
        state.output_activation = boundary_activation
        send_buffer, send_work = self.send_activation_async(boundary_activation.detach())
        state.activation_send_buffer = send_buffer
        state.activation_send_work = send_work
        return total_loss, total_auxiliary_loss

    def run_backward_action(
        self,
        *,
        runtime,
        states: list[PipelineMicroState],
        action: PipelineAction,
    ) -> None:
        context = runtime.state.step_context
        num_microbatches = runtime.plan.pp_schedule.microbatches
        if action.backward_step_idx == 0:
            status = PpStatus.BACKWARD_START
        elif action.backward_step_idx == num_microbatches - 1:
            status = PpStatus.BACKWARD_END
        else:
            status = PpStatus.BACKWARD_MIDDLE
        context.set_pp_state(microbatch_idx=action.microbatch_idx, status=status)
        state = states[action.microbatch_idx]
        if self.plugin.next_global_rank is None:
            assert state.raw_loss is not None
            runtime.state.loss = state.raw_loss / float(num_microbatches)
            DefaultStepRunner.run_backward(runtime)
            if self.plugin.prev_global_rank is not None:
                assert state.input_activation is not None and state.input_activation.grad is not None
                send_buffer, send_work = self.send_grad_async(state.input_activation.grad.detach())
                state.grad_send_buffer = send_buffer
                state.grad_send_work = send_work
            return

        assert state.output_activation is not None
        runtime.state.outputs = state.output_activation
        grad_output, recv_work = self.recv_grad_async(runtime, state.output_activation)
        recv_work.wait()
        if state.auxiliary_loss is None:
            DefaultStepRunner.run_backward(runtime, grad_output=grad_output)
        else:
            auxiliary_grad = torch.ones_like(state.auxiliary_loss)
            auxiliary_grad /= float(num_microbatches) * context.loss_divisor
            DefaultStepRunner.run_backward_many(
                runtime,
                tensors=[state.output_activation, state.auxiliary_loss],
                grad_tensors=[grad_output, auxiliary_grad],
            )
        if self.plugin.prev_global_rank is not None:
            assert state.input_activation is not None and state.input_activation.grad is not None
            send_buffer, send_work = self.send_grad_async(state.input_activation.grad.detach())
            state.grad_send_buffer = send_buffer
            state.grad_send_work = send_work

    def broadcast_loss(
        self,
        *,
        runtime,
        total_loss: torch.Tensor | None,
        total_auxiliary_loss: torch.Tensor | None,
        batch,
    ) -> torch.Tensor:
        device = model_device(runtime.model)
        if self.plugin.next_global_rank is None:
            assert total_loss is not None
            # Every PP rank participates in this collective.  The last stage's
            # loss may be bf16 under autocast while non-last stages have no loss
            # tensor at all; reduce a canonical fp32 metric value on every rank.
            # Backward has already consumed raw_loss, so this cast only affects
            # the returned/logged loss.
            loss = total_loss.float() / float(runtime.plan.pp_schedule.microbatches)
        else:
            loss = torch.zeros((), device=device, dtype=torch.float32)
        if self.plugin.pp_group is not None and self.plugin.stage_count > 1:
            all_reduce_tensor(loss, op=dist.ReduceOp.SUM, group=self.plugin.pp_group)
        if total_auxiliary_loss is None:
            return loss

        stage_auxiliary_loss = total_auxiliary_loss.float() / float(runtime.plan.pp_schedule.microbatches)
        global_auxiliary_loss = stage_auxiliary_loss.clone()
        if self.plugin.pp_group is not None and self.plugin.stage_count > 1:
            all_reduce_tensor(global_auxiliary_loss, op=dist.ReduceOp.SUM, group=self.plugin.pp_group)
        last_stage_auxiliary_loss = stage_auxiliary_loss if self.plugin.next_global_rank is None else torch.zeros_like(loss)
        if self.plugin.pp_group is not None and self.plugin.stage_count > 1:
            all_reduce_tensor(last_stage_auxiliary_loss, op=dist.ReduceOp.SUM, group=self.plugin.pp_group)
        ce_loss = loss - last_stage_auxiliary_loss
        total = ce_loss + global_auxiliary_loss
        coeff = getattr(runtime.model, "moe_aux_loss_coef", 0.0)
        runtime.state.metadata["model_metrics"] = {
            "loss/ce": float(ce_loss.item()),
            "moe/load_balance_loss": float((global_auxiliary_loss / coeff).item()) if coeff > 0 else 0.0,
            "loss/total": float(total.item()),
        }
        return total
        return loss

    def recv_activation_async(self, runtime, batch) -> tuple[torch.Tensor, object]:
        assert self.plugin.prev_global_rank is not None
        microbatch_idx = runtime.state.step_context.pp_cur_microbatch_idx
        device = model_device(runtime.model)
        dtype = activation_dtype(runtime)
        shape = activation_shape(
            batch,
            self.plugin.hidden_size,
            sequence_parallel_world_size=self.plugin.sequence_parallel_world_size,
        )
        buffer = acquire_buffer(
            shape=shape,
            dtype=dtype,
            device=device,
            policy=BufferPolicy.PINNED,
            key=f"pp.recv_activation.mb{microbatch_idx}",
        ).tensor
        work = irecv_tensor_async(buffer, self.plugin.prev_global_rank, group=self.plugin.pp_group)
        return buffer, work

    def send_activation_async(self, tensor: torch.Tensor) -> tuple[torch.Tensor, object]:
        assert self.plugin.next_global_rank is not None
        buffer = tensor.contiguous()
        work = isend_tensor_async(buffer, self.plugin.next_global_rank, group=self.plugin.pp_group)
        return buffer, work

    def recv_grad_async(self, runtime, output_activation: torch.Tensor) -> tuple[torch.Tensor, object]:
        assert self.plugin.next_global_rank is not None
        microbatch_idx = runtime.state.step_context.pp_cur_microbatch_idx
        grad = acquire_buffer(
            shape=tuple(output_activation.shape),
            dtype=output_activation.dtype,
            device=output_activation.device,
            policy=BufferPolicy.PINNED,
            key=f"pp.recv_grad.mb{microbatch_idx}",
        ).tensor
        work = irecv_tensor_async(grad, self.plugin.next_global_rank, group=self.plugin.pp_group)
        return grad, work

    def send_grad_async(self, tensor: torch.Tensor) -> tuple[torch.Tensor, object]:
        assert self.plugin.prev_global_rank is not None
        buffer = tensor.contiguous()
        work = isend_tensor_async(buffer, self.plugin.prev_global_rank, group=self.plugin.pp_group)
        return buffer, work


def split_batch(batch, num_microbatches: int):
    if num_microbatches == 1:
        return [batch]
    size = batch_size(batch)
    if size % num_microbatches != 0:
        raise ValueError(
            "pipeline microbatch split requires batch size divisible by microbatches, "
            f"got batch_size={size}, microbatches={num_microbatches}"
        )
    micro_batch_size = size // num_microbatches
    return [slice_batch(batch, index * micro_batch_size, micro_batch_size) for index in range(num_microbatches)]


def batch_size(batch) -> int:
    if isinstance(batch, dict):
        input_ids = batch.get(INPUT_IDS_KEY)
        if torch.is_tensor(input_ids):
            return int(input_ids.size(0))
    if isinstance(batch, (tuple, list)) and batch and torch.is_tensor(batch[0]):
        return int(batch[0].size(0))
    if torch.is_tensor(batch):
        return int(batch.size(0))
    raise TypeError(f"unsupported PP batch type={type(batch).__name__}")


def slice_batch(batch, start: int, length: int):
    if isinstance(batch, dict):
        return {
            key: value.narrow(0, start, length).contiguous() if torch.is_tensor(value) and value.size(0) >= start + length else value
            for key, value in batch.items()
        }
    if isinstance(batch, tuple):
        return tuple(slice_batch_item(value, start, length) for value in batch)
    if isinstance(batch, list):
        return [slice_batch_item(value, start, length) for value in batch]
    if torch.is_tensor(batch):
        return batch.narrow(0, start, length).contiguous()
    raise TypeError(f"unsupported PP batch type={type(batch).__name__}")


def slice_batch_item(value, start: int, length: int):
    if torch.is_tensor(value):
        return value.narrow(0, start, length).contiguous()
    return value


def unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, dict):
        return batch[INPUT_IDS_KEY], batch.get(LABELS_KEY)
    if isinstance(batch, (tuple, list)):
        if len(batch) != 2:
            raise ValueError(f"expected (input_ids, labels) batch tuple, got len={len(batch)}")
        return batch[0], batch[1]
    if torch.is_tensor(batch):
        return batch, None
    raise TypeError(f"unsupported PP batch type={type(batch).__name__}")


def extract_labels(batch) -> torch.Tensor | None:
    _input_ids, labels = unpack_batch(batch)
    return labels


def activation_shape(
    batch,
    hidden_size: int,
    *,
    sequence_parallel_world_size: int = 1,
) -> tuple[int, int, int]:
    input_ids, _labels = unpack_batch(batch)
    seq_len = int(input_ids.size(1))
    if sequence_parallel_world_size > 1:
        if seq_len % sequence_parallel_world_size != 0:
            raise ValueError(
                "PP+SP activation shape requires sequence length divisible by tp world size, "
                f"got seq_len={seq_len}, tp={sequence_parallel_world_size}"
            )
        seq_len //= sequence_parallel_world_size
    return int(input_ids.size(0)), seq_len, hidden_size


def model_device(model: nn.Module) -> torch.device:
    first_param = next(model.parameters(), None)
    if first_param is None:
        raise ValueError("PP model must have parameters")
    return first_param.device


def activation_dtype(runtime) -> torch.dtype:
    if runtime.dtype is not None:
        return runtime.dtype
    first_param = next(runtime.model.parameters(), None)
    if first_param is None:
        raise ValueError("PP model must have parameters")
    return first_param.dtype


def cast_boundary_activation(tensor: torch.Tensor, runtime) -> torch.Tensor:
    target_dtype = activation_dtype(runtime)
    if tensor.dtype == target_dtype:
        return tensor
    return tensor.to(dtype=target_dtype)
