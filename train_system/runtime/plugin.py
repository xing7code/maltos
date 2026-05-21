from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable
import torch.nn as nn

from train_system.parallel.specs import TpSpParallelSpec

if TYPE_CHECKING:
    from train_system.runtime.core import RuntimeCore, RuntimePhase


@runtime_checkable
class ParallelizableModule(Protocol):
    def parallelize_spec(self) -> TpSpParallelSpec: ...


class PluginId(str, Enum):
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
    PROFILER = "profiler"



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
