from __future__ import annotations

from typing import Protocol, runtime_checkable
import torch.nn as nn

from train_system.parallel.spec import ModelParallelSpec


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


