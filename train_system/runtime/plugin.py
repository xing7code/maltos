from __future__ import annotations

import torch.nn as nn


class BaseParallelPlugin:
    """Composable parallel runtime hook.

    Each plugin owns one concern (DP/TP/PP/CP/EP/ZeRO) and mutates runtime state
    through explicit lifecycle callbacks.
    """

    def setup_model(self, model: nn.Module) -> nn.Module:
        return model

    def before_forward(self) -> None:
        pass

    def after_backward(self) -> None:
        pass

    def step(self) -> None:
        pass


