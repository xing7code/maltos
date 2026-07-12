from __future__ import annotations

import torch
import torch.nn as nn

from runtime.plugin import PluginId, RuntimePlugin
from runtime.types import MetricValue, RuntimePhase


class _AutocastModelWrapper(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype, device_type: str):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.device_type = device_type

    def forward(self, *args, **kwargs):
        with torch.autocast(device_type=self.device_type, dtype=self.compute_dtype):
            return self.module(*args, **kwargs)

    def flops_per_token(self) -> float:
        if not hasattr(self.module, "flops_per_token"):
            raise AttributeError("wrapped module does not define flops_per_token")
        return float(self.module.flops_per_token())


class PrecisionPlugin(RuntimePlugin):
    def __init__(
        self,
        compute_dtype: torch.dtype | None = None,
        use_grad_scaler: bool = True,
    ) -> None:
        super().__init__(
            id=PluginId.PRECISION,
            name="precision",
            runs_after={
                PluginId.TP,
                PluginId.SP,
                PluginId.DP,
                PluginId.PP,
                PluginId.CP,
                PluginId.EP,
                PluginId.ZERO1,
                PluginId.ZERO2,
                PluginId.ZERO3,
            },
        )
        self.compute_dtype = compute_dtype
        self.use_grad_scaler = use_grad_scaler

    def transform_model(self, model: nn.Module) -> nn.Module:
        self._setup_scaler()
        if self.compute_dtype is None:
            return model
        _validate_compute_dtype(self.compute_dtype)
        assert self.runtime is not None
        device = torch.device(self.runtime.device)
        return _AutocastModelWrapper(model, self.compute_dtype, device.type)

    def on_phase(self, phase: RuntimePhase) -> None:
        if self.runtime is None:
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
        device = torch.device(self.runtime.device)
        if device.type != "cuda":
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

    def collect_metrics(self) -> dict[str, MetricValue]:
        if self.runtime is None:
            return {}
        return {
            "compute_dtype": None if self.compute_dtype is None else str(self.compute_dtype),
            "loss_scale": self.runtime.state.metadata.get("loss_scale"),
            "overflow": bool(self.runtime.state.metadata.get("overflow", False)),
        }


def _validate_compute_dtype(compute_dtype: torch.dtype) -> None:
    if compute_dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError(
            f"PrecisionPlugin only supports compute_dtype in {{torch.float16, torch.bfloat16}}, got {compute_dtype}"
        )
