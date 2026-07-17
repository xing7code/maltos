from __future__ import annotations

import time

import torch

from runtime.plugin import PluginId, RuntimePlugin
from runtime.types import MetricValue, RuntimePhase


class MetricPlugin(RuntimePlugin):
    def __init__(self, include_cuda_memory: bool = True) -> None:
        super().__init__(
            id=PluginId.METRICS,
            name="metrics",
            runs_after={PluginId.FP16, PluginId.GRAD_CLIP},
        )
        self.include_cuda_memory = include_cuda_memory
        self._step_start: float | None = None
        self._metrics: dict[str, MetricValue] = {}

    def on_phase(self, phase: RuntimePhase) -> None:
        if phase == RuntimePhase.PRE_STEP_RUNNER:
            assert self.runtime is not None
            context = self.runtime.state.step_context
            if context is None or context.accum_start:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                self._step_start = time.perf_counter()
            return
        if phase == RuntimePhase.PRE_STEP:
            return
        if phase == RuntimePhase.POST_STEP:
            if self._step_start is not None:
                self._metrics["perf/step_sec"] = time.perf_counter() - self._step_start
                self._step_start = None
            if self.include_cuda_memory:
                self._metrics.update(_cuda_memory_metrics())

    def collect_metrics(self) -> dict[str, MetricValue]:
        return dict(self._metrics)


def _cuda_memory_metrics() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    device = torch.cuda.current_device()
    return {
        "memory/allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "memory/reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "memory/max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "memory/max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
    }
