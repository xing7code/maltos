from __future__ import annotations

import torch

from train_system.runtime.core import RuntimePhase
from train_system.runtime.plugin import PluginId, RuntimePlugin


class GradClipPlugin(RuntimePlugin):
    def __init__(self, max_norm: float, record_grad_norm: bool = True) -> None:
        super().__init__(id=PluginId.GRAD_CLIP, name="grad_clip", runs_after={PluginId.PRECISION})
        self.max_norm = max_norm
        self.record_grad_norm = record_grad_norm

    def on_phase(self, phase: RuntimePhase) -> None:
        if phase != RuntimePhase.PRE_STEP or self.runtime is None:
            return
        optimizer, _ = self.runtime.get_optimizer_and_scheduler()
        if optimizer is None:
            return

        scaler = self.runtime.state.scaler
        if scaler is not None:
            scaler.unscale_(optimizer)

        grad_params = []
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param is not None and param.grad is not None:
                    grad_params.append(param)
        if not grad_params:
            return
        norm = torch.nn.utils.clip_grad_norm_(grad_params, self.max_norm)
        if self.record_grad_norm:
            self.runtime.state.metadata["grad_norm"] = float(norm.item())
