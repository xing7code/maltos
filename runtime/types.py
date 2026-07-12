from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


MetricValue = float | int | str | bool | None


class RuntimePhase(str, Enum):
    PRE_STEP_RUNNER = "pre_step_runner"
    PRE_FORWARD = "pre_forward"
    POST_FORWARD = "post_forward"
    PRE_BACKWARD = "pre_backward"
    POST_BACKWARD = "post_backward"
    PRE_STEP = "pre_step"
    POST_STEP = "post_step"
    PRE_SAVE = "pre_save"
    POST_LOAD = "post_load"


class ParamRole(str, Enum):
    SHARED = "shared"
    EXPERT = "expert"


class PpStatus(str, Enum):
    IDLE = "idle"
    FORWARD = "forward"
    BACKWARD_START = "backward_start"
    BACKWARD_MIDDLE = "backward_middle"
    BACKWARD_END = "backward_end"


@dataclass
class StepContext:
    step: int = 0
    microbatch_idx: int = 0
    grad_accum_steps: int = 1
    pp_cur_microbatch_idx: int = 0
    pp_status: PpStatus = PpStatus.IDLE

    def __post_init__(self) -> None:
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")

    @property
    def accum_start(self) -> bool:
        return self.microbatch_idx == 0 and self.pp_status in {PpStatus.IDLE, PpStatus.BACKWARD_START}

    @property
    def is_step_boundary(self) -> bool:
        return (
            ((self.microbatch_idx + 1) % self.grad_accum_steps) == 0
            and self.pp_status in {PpStatus.IDLE, PpStatus.BACKWARD_END}
        )

    @property
    def backward_start(self) -> bool:
        """True at the first backward of each grad-accum micro-step.

        Unlike ``accum_start`` (gated on ``microbatch_idx == 0``, so False for
        every accum step after the first), this fires once per micro-step's
        backward phase. Per-micro-step reduction counters must be re-armed here,
        not at ``accum_start``: on the step-boundary micro-step (microbatch_idx
        != 0 when grad_accum > 1) the counter would otherwise never be reset and
        the async grad-reduce worker reads a stale zero, firing before backward
        has produced its handles.
        """
        return self.pp_status in {PpStatus.IDLE, PpStatus.BACKWARD_START}

    @property
    def loss_divisor(self) -> float:
        return float(self.grad_accum_steps)

    def set_pp_state(self, *, microbatch_idx: int, status: PpStatus) -> None:
        if microbatch_idx < 0:
            raise ValueError(f"pp_cur_microbatch_idx must be >= 0, got {microbatch_idx}")
        self.pp_cur_microbatch_idx = microbatch_idx
        self.pp_status = status

    def advance_micro_step(self) -> bool:
        should_step = self.is_step_boundary
        self.microbatch_idx = (self.microbatch_idx + 1) % self.grad_accum_steps
        self.pp_cur_microbatch_idx = 0
        self.pp_status = PpStatus.IDLE
        return should_step

    def advance_step(self) -> None:
        self.step += 1


@dataclass
class RuntimeState:
    """Transient execution context for the current step.

    This is intentionally not a checkpoint manifest. Plugins use it as a
    shared scratchpad for phase-local data such as the current batch, loss,
    microbatch index, profiler annotations, or temporary scheduling metadata.
    Durable training state should be declared through checkpoint/state APIs.
    """

    step_context: StepContext = field(default_factory=StepContext)
    loss: torch.Tensor | None = None
    batch: Any = None
    outputs: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    static_metrics: dict[str, MetricValue] = field(default_factory=dict)
    scaler: torch.amp.GradScaler | None = None
    omitted_module_paths: set[str] = field(default_factory=set)

    @property
    def step(self) -> int:
        return self.step_context.step

    @step.setter
    def step(self, value: int) -> None:
        self.step_context.step = value
