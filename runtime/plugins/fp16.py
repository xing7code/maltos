from __future__ import annotations

import torch
import torch.nn as nn

from runtime.plugin import PluginId, RuntimePlugin
from runtime.types import MetricValue, RuntimePhase, SetupPhase


class Fp16Plugin(RuntimePlugin):
    def __init__(self) -> None:
        super().__init__(
            id=PluginId.FP16,
            name="fp16",
            runs_after={
                PluginId.TP,
                PluginId.SP,
                PluginId.DP,
                PluginId.PP,
                PluginId.CP,
                PluginId.EP,
            },
        )

    def on_setup_phase(self, phase: SetupPhase, model: nn.Module) -> nn.Module:
        if phase == SetupPhase.FINALIZE:
            self._setup_scaler()
        return model

    def on_step_phase(self, phase: RuntimePhase) -> None:
        if self.runtime is None:
            return
        if phase == RuntimePhase.PRE_BACKWARD:
            self._scale_loss_for_backward()

    def _setup_scaler(self) -> None:
        if self.runtime is None:
            return
        if self.runtime.dtype != torch.float16:
            self.runtime.state.scaler = None
            return
        device = torch.device(self.runtime.device)
        if device.type != "cuda":
            raise ValueError("Fp16Plugin requires CUDA model parameters")
        self.runtime.state.scaler = torch.amp.GradScaler(device="cuda")

    def _scale_loss_for_backward(self) -> None:
        if self.runtime is None or self.runtime.state.scaler is None:
            return
        if self.runtime.state.loss is None:
            return
        self.runtime.state.loss = self.runtime.state.scaler.scale(self.runtime.state.loss)

    def export_plugin_state(self) -> dict[str, object]:
        if self.runtime is None or self.runtime.state.scaler is None:
            return {}
        return {"scaler": self.runtime.state.scaler.state_dict()}

    def import_plugin_state(self, state: dict[str, object]) -> None:
        if self.runtime is None or self.runtime.state.scaler is None:
            return
        scaler_state = state.get("scaler")
        if isinstance(scaler_state, dict):
            self.runtime.state.scaler.load_state_dict(scaler_state)

    def collect_metrics(self) -> dict[str, MetricValue]:
        if self.runtime is None:
            return {}
        return {
            "dtype": None if self.runtime.dtype is None else str(self.runtime.dtype),
            "loss_scale": self.runtime.state.metadata.get("loss_scale"),
            "overflow": bool(self.runtime.state.metadata.get("overflow", False)),
        }
