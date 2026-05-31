from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable
import torch
import torch.nn as nn

from parallel.specs import TpSpParallelSpec
from runtime.mesh import MeshAxis
from state.state import ParamState

if TYPE_CHECKING:
    from runtime.core import RuntimeCore, RuntimePhase


MetricValue = float | int | str | bool | None


@runtime_checkable
class ParallelizableModule(Protocol):
    def parallelize_spec(self) -> TpSpParallelSpec: ...


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
    runtime: "RuntimeCore | None" = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = self.id.value

    def bind(self, runtime: "RuntimeCore") -> None:
        self.runtime = runtime

    def transform_model(self, model: nn.Module) -> nn.Module:
        return model

    def on_phase(self, phase: "RuntimePhase") -> None:
        pass

    def runtime_optimizer_replicated_axes(self) -> set[MeshAxis]:
        return set()

    def runtime_optimizer_sharded_axes(self) -> set[MeshAxis]:
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
