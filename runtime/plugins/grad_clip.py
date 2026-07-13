from __future__ import annotations

import torch
import torch.distributed as dist

from runtime.plugin import PluginId, RuntimePlugin
from runtime.types import MetricValue, RuntimePhase


class GradClipPlugin(RuntimePlugin):
    def __init__(self, max_norm: float) -> None:
        super().__init__(
            id=PluginId.GRAD_CLIP,
            name="grad_clip",
            # Must run after ZeRO variants: their PRE_STEP calls _wait_grad_sync(),
            # which joins the async reduction thread and populates local_param.grad.
            # Without this ordering, we'd read stale / unset gradients on nccl.
            runs_after={PluginId.PRECISION, PluginId.ZERO1, PluginId.ZERO2, PluginId.ZERO3},
        )
        self.max_norm = max_norm

    def bind(self, runtime) -> None:
        super().bind(runtime)
        if any(plugin.id in {PluginId.ZERO1, PluginId.ZERO2, PluginId.ZERO3} for plugin in runtime.plugins):
            raise ValueError(
                "GradClipPlugin only supports runtime-owned optimizers. "
                "For ZeRO, set RuntimeCore.grad_clip_max_norm and let the ZeRO plugin clip local shards."
            )

    def on_phase(self, phase: RuntimePhase) -> None:
        if phase != RuntimePhase.PRE_STEP or self.runtime is None:
            return
        optimizer, _ = self.runtime.get_optimizer_and_scheduler()
        if optimizer is None:
            return

        scaler = self.runtime.state.scaler
        if scaler is not None:
            scaler.unscale_(optimizer)

        grad_params = [
            param
            for group in optimizer.param_groups
            for param in group["params"]
            if param is not None and param.grad is not None
        ]
        if not grad_params:
            return

        global_norm = self._global_grad_norm(grad_params)
        clip_coef = float(self.max_norm) / (global_norm + 1e-6)
        if clip_coef < 1.0:
            for param in grad_params:
                param.grad.detach().mul_(clip_coef)

        self.runtime.state.metadata["grad_norm"] = global_norm

    def _global_grad_norm(self, params: list) -> float:
        assert self.runtime is not None
        device = params[0].grad.device
        local_sq = torch.zeros((), dtype=torch.float32, device=device)
        for p in params:
            if p.grad is None:
                continue
            replica_factor = float(self.runtime.grad_norm_replica_factor(p))
            local_sq.add_(p.grad.detach().float().pow(2).sum() / replica_factor)

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)

        return float(local_sq.sqrt().item())

    def collect_metrics(self) -> dict[str, MetricValue]:
        if self.runtime is None:
            return {}
        grad_norm = self.runtime.state.metadata.get("grad_norm")
        if grad_norm is None:
            return {}
        return {"grad_norm": float(grad_norm), "max_norm": float(self.max_norm)}
