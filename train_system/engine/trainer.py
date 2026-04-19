from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from train_system.runtime.context import RuntimeContext


@dataclass
class Trainer:
    context: RuntimeContext
    model: nn.Module
    optimizer: torch.optim.Optimizer

    def setup(self) -> None:
        for plugin in self.context.plugins:
            self.model = plugin.setup_model(self.model)

    def train_steps(self, data: Iterable, steps: int) -> None:
        self.model.train()
        iterator = iter(data)

        for _ in range(steps):
            batch = next(iterator)
            for plugin in self.context.plugins:
                plugin.before_forward(self.model)

            loss = self.model(batch)
            loss.backward()

            for plugin in self.context.plugins:
                plugin.after_backward(self.model)

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            for plugin in self.context.plugins:
                plugin.step(self.model)
