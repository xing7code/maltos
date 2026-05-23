from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from train_system.runtime.core import RuntimePhase
from train_system.runtime.plugin import PluginId, RuntimePlugin

if TYPE_CHECKING:
    from train_system.runtime.core import RuntimeCore


class _AutocastModelWrapper(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype, device_type: str):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.device_type = device_type

    def forward(self, *args, **kwargs):
        with torch.autocast(device_type=self.device_type, dtype=self.compute_dtype):
            return self.module(*args, **kwargs)


class PrecisionPlugin(RuntimePlugin):
    def __init__(
        self,
        compute_dtype: torch.dtype | None = None,
        use_grad_scaler: bool = True,
    ) -> None:
        super().__init__(
            id=PluginId.PRECISION,
            name="precision",
            runs_after={PluginId.TP, PluginId.SP, PluginId.ZERO1, PluginId.ZERO2, PluginId.ZERO3},
        )
        self.compute_dtype = compute_dtype
        self.use_grad_scaler = use_grad_scaler

    def transform_model(self, model: nn.Module) -> nn.Module:
        if self.compute_dtype is None:
            return model
        _validate_compute_dtype(self.compute_dtype)
        return _AutocastModelWrapper(model, self.compute_dtype, _model_device_type(model))

    def on_phase(self, phase: RuntimePhase) -> None:
        if self.runtime is None:
            return
        if phase == RuntimePhase.SETUP:
            self._setup_scaler()
            return
        if phase == RuntimePhase.PRE_BACKWARD:
            self._scale_loss_for_backward()
            return

    def _setup_scaler(self) -> None:
        if self.runtime is None:
            return
        if self.compute_dtype != torch.float16 or not self.use_grad_scaler:
            self.runtime.state.scaler = None
            return
        device_type = _model_device_type(self.runtime.model)
        if device_type != "cuda":
            raise ValueError("PrecisionPlugin(fp16) requires CUDA model parameters")
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


def _validate_compute_dtype(compute_dtype: torch.dtype) -> None:
    if compute_dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(
            f"PrecisionPlugin only supports compute_dtype in {{torch.float16, torch.bfloat16}}, got {compute_dtype}"
        )


def _model_device_type(model: nn.Module) -> str:
    first_param = next(model.parameters(), None)
    return "cpu" if first_param is None else first_param.device.type
