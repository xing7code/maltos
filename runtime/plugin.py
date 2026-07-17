from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING
import torch
import torch.nn as nn

from parallel.protocols import (
    ContextParallelizableModule,
    ExpertParallelizableModule,
    FlopsEstimatableModule,
    PipelineParallelizableModule,
    TpSpParallelizableModule,
)
if TYPE_CHECKING:
    from runtime.core import RuntimeCore
    from runtime.step_runners import StepRunner
    from runtime.types import RuntimePhase

from runtime.types import MetricValue

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
    runtime: "RuntimeCore | None" = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.id.value
        if self.owns_optimizer and type(self).optimizer_state_source_rank is RuntimePlugin.optimizer_state_source_rank:
            raise ValueError(
                f"plugin={self.id.value} owns_optimizer=True must implement optimizer_state_source_rank()"
            )

    def bind(self, runtime: "RuntimeCore") -> None:
        self.runtime = runtime

    def transform_model(self, model: nn.Module) -> nn.Module:
        return model

    def annotate_param_metadata(self) -> None:
        """Record parameter-level runtime metadata after transform_model()."""
        pass

    def on_phase(self, phase: "RuntimePhase") -> None:
        pass

    def build_step_runner(self) -> "StepRunner | None":
        return None

    def optimizer_state_source_rank(self, rank_id: int) -> int:
        raise NotImplementedError(
            f"plugin={self.id.value} owns_optimizer={self.owns_optimizer} must implement optimizer_state_source_rank()"
        )

    def override_param_state_dict(self) -> tuple[dict[str, torch.Tensor], list[ParamState]] | None:
        return None

    def load_param_state_dict(self, state: dict[str, torch.Tensor]) -> bool:
        return False

    def export_plugin_state(self) -> dict[str, object]:
        return {}

    def import_plugin_state(self, state: dict[str, object]) -> None:
        pass

    def collect_metrics(self) -> dict[str, MetricValue]:
        return {}

    def close(self) -> None:
        pass
