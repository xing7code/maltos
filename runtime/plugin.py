from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable
import torch
import torch.nn as nn

from parallel.context import ContextParallelSpec
from parallel.expert import ExpertParallelSpec
from parallel.pipeline import PipelineParallelSpec
from parallel.specs import TpSpParallelSpec
from runtime.mesh import MeshAxis
from state.state import ParamState

if TYPE_CHECKING:
    from runtime.core import RuntimeCore
    from runtime.step_runners import StepRunner
    from runtime.types import RuntimePhase

from runtime.types import MetricValue


@runtime_checkable
class TpSpParallelizableModule(Protocol):
    def tpsp_parallelize_spec(self) -> TpSpParallelSpec: ...


@runtime_checkable
class PipelineParallelizableModule(Protocol):
    def pipeline_parallel_spec(self) -> PipelineParallelSpec: ...


@runtime_checkable
class ContextParallelizableModule(Protocol):
    def context_parallel_spec(self) -> ContextParallelSpec: ...


@runtime_checkable
class ExpertParallelizableModule(Protocol):
    def expert_parallel_spec(self) -> ExpertParallelSpec: ...


@runtime_checkable
class FlopsEstimatableModule(Protocol):
    def flops_per_token(self) -> float: ...


class PluginId(str, Enum):
    PRECISION = "precision"
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
    PERF_METRICS = "perf_metrics"
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

    def bind(self, runtime: "RuntimeCore") -> None:
        self.runtime = runtime

    def transform_model(self, model: nn.Module) -> nn.Module:
        return model

    def annotate_param_layout(self) -> None:
        """Record parameter-level distributed layout metadata after transform_model()."""
        pass

    def on_phase(self, phase: "RuntimePhase") -> None:
        pass

    def build_step_runner(self) -> "StepRunner | None":
        return None

    def runtime_optimizer_replicated_axes(self) -> set[MeshAxis]:
        """Optimizer-level replicated axes for source-rank/checkpoint ownership."""
        return set()

    def runtime_optimizer_sharded_axes(self) -> set[MeshAxis]:
        """Optimizer-level sharded axes for source-rank/checkpoint ownership."""
        return set()

    def optimizer_state_source_rank(self, rank_id: int) -> int:
        return rank_id

    def override_param_state_dict(self) -> tuple[dict[str, torch.Tensor], list[ParamState]] | None:
        return None

    def load_param_state_dict(self, state: dict[str, torch.Tensor]) -> bool:
        return False

    def annotate_checkpoint_state(self, entry: ParamState) -> None:
        pass

    def export_plugin_state(self) -> dict[str, object]:
        return {}

    def import_plugin_state(self, state: dict[str, object]) -> None:
        pass

    def collect_metrics(self) -> dict[str, MetricValue]:
        return {}

    def close(self) -> None:
        pass
