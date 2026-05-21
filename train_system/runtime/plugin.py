from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable
import torch.nn as nn

from train_system.parallel.spec import ModelParallelSpec

if TYPE_CHECKING:
    from train_system.runtime.core import RuntimeCore, RuntimePhase


@runtime_checkable
class ParallelizableModule(Protocol):
    def parallelize_spec(self) -> ModelParallelSpec: ...



class BaseParallelPlugin:
    """Composable parallel runtime hook.

    Each plugin owns one concern (DP/TP/PP/CP/EP/ZeRO) and mutates runtime state
    through explicit lifecycle callbacks.
    """

    def setup_model(self, model: nn.Module) -> nn.Module:
        return model

    def before_forward(self, model: nn.Module) -> None:
        pass

    def after_backward(self, model: nn.Module) -> None:
        pass

    def step(self, model: nn.Module) -> None:
        pass


@dataclass
class RuntimePlugin:
    """Draft plugin contract for the next runtime core."""

    name: str
    requires: set[str] = field(default_factory=set)
    runs_after: set[str] = field(default_factory=set)
    runs_before: set[str] = field(default_factory=set)
    runtime: "RuntimeCore | None" = field(default=None, init=False, repr=False)

    def bind(self, runtime: "RuntimeCore") -> None:
        self.runtime = runtime

    def transform_model(self, model: nn.Module) -> nn.Module:
        return model

    def on_phase(self, phase: "RuntimePhase") -> None:
        pass

