from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import torch

from runtime.types import RuntimePhase

if TYPE_CHECKING:
    from runtime.core import RuntimeCore


class StepRunner(Protocol):
    def run(self, runtime: "RuntimeCore", batch: Any) -> torch.Tensor: ...


class DefaultStepRunner:
    def run(self, runtime: "RuntimeCore", batch: Any) -> torch.Tensor:
        self.run_forward(runtime, batch)
        if not torch.is_tensor(runtime.state.loss):
            raise TypeError("RuntimeCore expects model(batch) to return a Tensor loss during training.")
        self.run_backward(runtime)
        assert runtime.state.loss is not None
        return runtime.state.loss

    @staticmethod
    def run_forward(runtime: "RuntimeCore", batch: Any) -> None:
        runtime._run_phase(RuntimePhase.PRE_FORWARD)
        try:
            outputs = runtime.model(batch)
            runtime.state.outputs = outputs
            runtime.state.loss = outputs if torch.is_tensor(outputs) else None
        finally:
            runtime._run_phase(RuntimePhase.POST_FORWARD)

    @staticmethod
    def run_backward(
        runtime: "RuntimeCore",
        *,
        grad_output: torch.Tensor | None = None,
    ) -> None:
        runtime._run_phase(RuntimePhase.PRE_BACKWARD)
        if grad_output is None:
            if runtime.state.loss is None:
                raise TypeError("RuntimeCore expected runtime.state.loss to be a Tensor before backward()")
            divisor = runtime.state.step_context.loss_divisor
            if divisor != 1:
                runtime.state.loss = runtime.state.loss / divisor
            runtime.state.loss.backward()
        else:
            if not torch.is_tensor(runtime.state.outputs):
                raise TypeError("RuntimeCore expected runtime.state.outputs Tensor for activation backward()")
            runtime.state.outputs.backward(grad_output)
        runtime._run_phase(RuntimePhase.POST_BACKWARD)
