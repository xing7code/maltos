from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import torch

from runtime.types import LossOutput, PipelineOutput, RuntimePhase
from utils.constants import IGNORE_INDEX, LABELS_KEY
from utils.profiling import profiled

if TYPE_CHECKING:
    from runtime.core import RuntimeCore


class StepRunner(Protocol):
    def run(self, runtime: "RuntimeCore", batch: Any) -> torch.Tensor: ...


def count_label_tokens(batch: Any) -> int | None:
    """Number of supervised targets in a micro-batch, or None if unknown."""
    labels = None
    if isinstance(batch, dict):
        labels = batch.get(LABELS_KEY)
    elif isinstance(batch, (tuple, list)) and len(batch) > 1:
        labels = batch[1]
    if not torch.is_tensor(labels):
        return None
    return int(labels.ne(IGNORE_INDEX).sum().item())


class DefaultStepRunner:
    def run(self, runtime: "RuntimeCore", batch: Any) -> torch.Tensor:
        context = runtime.state.step_context
        if context.token_weighted_loss:
            count = count_label_tokens(batch)
            if count is None:
                raise ValueError(
                    "token-weighted loss needs labels in the batch to count supervised targets"
                )
            runtime.state.metadata["label_tokens_this_microbatch"] = count
            context.add_label_tokens(count)
        self.run_forward(runtime, batch)
        if not torch.is_tensor(runtime.state.loss):
            raise TypeError("RuntimeCore expects model(batch) to return a Tensor loss during training.")
        self.run_backward(runtime)
        assert runtime.state.loss is not None
        return runtime.state.loss

    @staticmethod
    @profiled("maltos::forward")
    def run_forward(runtime: "RuntimeCore", batch: Any) -> None:
        runtime._run_step_phase(RuntimePhase.PRE_FORWARD)
        try:
            outputs = runtime.model(batch)
            runtime.state.outputs = outputs
            if isinstance(outputs, LossOutput):
                runtime.state.loss = outputs.loss
                runtime.state.metadata["model_metrics"] = dict(outputs.metrics)
            elif isinstance(outputs, PipelineOutput):
                runtime.state.loss = None
                runtime.state.metadata.pop("model_metrics", None)
            else:
                runtime.state.loss = outputs if torch.is_tensor(outputs) else None
                runtime.state.metadata.pop("model_metrics", None)
        finally:
            runtime._run_step_phase(RuntimePhase.POST_FORWARD)

    @staticmethod
    @profiled("maltos::backward")
    def run_backward(
        runtime: "RuntimeCore",
        *,
        grad_output: torch.Tensor | None = None,
    ) -> None:
        runtime._run_step_phase(RuntimePhase.PRE_BACKWARD)
        if grad_output is None:
            if runtime.state.loss is None:
                raise TypeError("RuntimeCore expected runtime.state.loss to be a Tensor before backward()")
            # Keep the unscaled loss for telemetry.  ``state.loss`` is divided
            # by grad accumulation below, so reading it after backward reports
            # a misleading loss/grad_accum value.
            runtime.state.metadata["raw_loss_for_metrics"] = runtime.state.loss.detach().float()
            context = runtime.state.step_context
            if context.token_weighted_loss:
                # The model returns a mean over this micro-batch's own targets,
                # so multiplying by that count recovers the summed CE.  Dividing
                # by a fixed nominal keeps gradient magnitudes in their usual
                # range; the step's real normaliser is applied once all
                # micro-batches and ranks have contributed.
                count = runtime.state.metadata.get("label_tokens_this_microbatch")
                if count is None:
                    raise ValueError("token-weighted loss lost its per-micro-batch target count")
                scale = float(count) / float(context.nominal_tokens_per_microbatch)
                runtime.state.loss = runtime.state.loss * scale
            else:
                divisor = context.loss_divisor
                if divisor != 1:
                    runtime.state.loss = runtime.state.loss / divisor
            runtime.state.loss.backward()
        else:
            if not torch.is_tensor(runtime.state.outputs):
                raise TypeError("RuntimeCore expected runtime.state.outputs Tensor for activation backward()")
            runtime.state.outputs.backward(grad_output)
        runtime._run_step_phase(RuntimePhase.POST_BACKWARD)

    @staticmethod
    @profiled("maltos::backward")
    def run_backward_many(
        runtime: "RuntimeCore",
        tensors: list[torch.Tensor],
        grad_tensors: list[torch.Tensor],
    ) -> None:
        """Run one autograd traversal for activation and auxiliary gradients."""
        if len(tensors) != len(grad_tensors):
            raise ValueError("backward tensors and grad_tensors must have the same length")
        runtime._run_step_phase(RuntimePhase.PRE_BACKWARD)
        torch.autograd.backward(tensors=tensors, grad_tensors=grad_tensors)
        runtime._run_step_phase(RuntimePhase.POST_BACKWARD)
