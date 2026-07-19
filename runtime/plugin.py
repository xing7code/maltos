from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
import torch.nn as nn

from parallel.protocols import (
    ContextParallelizableModule,
    ExpertParallelizableModule,
    FlopsEstimatableModule,
    PipelineParallelizableModule,
    TpSpParallelizableModule,
)
from runtime.types import MetricValue, SetupPhase

if TYPE_CHECKING:
    from runtime.core import RuntimeCore
    from runtime.step_runners import StepRunner
    from runtime.types import RuntimePhase, SetupPhase
    from state.state import ParamState, StateManager


class PluginId(str, Enum):
    FP16 = "fp16"
    GRAD_CLIP = "grad_clip"
    TP = "tp"
    SP = "sp"
    TP_SP = "tp_sp"
    DP = "dp"
    PP = "pp"
    CP = "cp"
    EP = "ep"
    ZERO1 = "zero1"
    ZERO2 = "zero2"
    ZERO3 = "zero3"
    CHECKPOINT = "checkpoint"
    METRICS = "metrics"
    TORCH_PROFILER = "torch_profiler"


@runtime_checkable
class OptimizerOwner(Protocol):
    def get_optimizer_and_scheduler(
        self,
    ) -> tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LRScheduler | None]: ...

    def optimizer_state_source_rank(self, rank_id: int) -> int: ...


@runtime_checkable
class StepRunnerOwner(Protocol):
    def build_step_runner(self) -> "StepRunner | None": ...


@runtime_checkable
class ModelStateOwner(Protocol):
    def export_model_state(
        self,
        state_manager: "StateManager",
    ) -> tuple[dict[str, torch.Tensor], list["ParamState"]]: ...

    def import_model_state(
        self,
        state_manager: "StateManager",
        model_state: dict[str, torch.Tensor],
    ) -> None: ...


@dataclass
class RuntimePlugin:
    """Draft plugin contract for the next runtime core."""

    id: PluginId
    name: str | None = None
    requires: set[PluginId] = field(default_factory=set)
    runs_after: set[PluginId] = field(default_factory=set)
    runs_before: set[PluginId] = field(default_factory=set)
    owns_optimizer: bool = False
    owns_step_runner: bool = False
    owns_model_state: bool = False
    runtime: "RuntimeCore | None" = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.id.value
        if self.owns_optimizer and not isinstance(self, OptimizerOwner):
            raise ValueError(
                f"plugin={self.id.value} owns_optimizer=True must satisfy OptimizerOwner"
            )
        if self.owns_step_runner and not isinstance(self, StepRunnerOwner):
            raise ValueError(
                f"plugin={self.id.value} owns_step_runner=True must satisfy StepRunnerOwner"
            )
        if self.owns_model_state and not isinstance(self, ModelStateOwner):
            raise ValueError(
                f"plugin={self.id.value} owns_model_state=True must satisfy ModelStateOwner"
            )

    def bind(self, runtime: "RuntimeCore") -> None:
        self.runtime = runtime

    def on_setup_phase(self, phase: "SetupPhase", model: nn.Module) -> nn.Module:
        return model

    def annotate_param_metadata(self) -> None:
        """Record parameter-level runtime metadata after setup transforms."""
        pass

    def on_step_phase(self, phase: "RuntimePhase") -> None:
        pass

    def build_step_runner(self) -> "StepRunner | None":
        return None

    def get_optimizer_and_scheduler(
        self,
    ) -> tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LRScheduler | None]:
        return getattr(self, "optimizer", None), getattr(self, "scheduler", None)

    def optimizer_state_source_rank(self, rank_id: int) -> int:
        raise NotImplementedError(
            f"plugin={self.id.value} owns_optimizer={self.owns_optimizer} must implement optimizer_state_source_rank()"
        )

    def export_model_state(
        self,
        state_manager: "StateManager",
    ) -> tuple[dict[str, torch.Tensor], list["ParamState"]]:
        raise NotImplementedError(
            f"plugin={self.id.value} owns_model_state={self.owns_model_state} must implement export_model_state()"
        )

    def import_model_state(
        self,
        state_manager: "StateManager",
        model_state: dict[str, torch.Tensor],
    ) -> None:
        raise NotImplementedError(
            f"plugin={self.id.value} owns_model_state={self.owns_model_state} must implement import_model_state()"
        )

    def export_plugin_state(self) -> dict[str, object]:
        return {}

    def import_plugin_state(self, state: dict[str, object]) -> None:
        pass

    def collect_metrics(self) -> dict[str, MetricValue]:
        return {}

    def close(self) -> None:
        pass
