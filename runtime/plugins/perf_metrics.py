from __future__ import annotations

import time

import torch

from runtime.core import RuntimePhase
from runtime.plugin import MetricValue, PluginId, RuntimePlugin


class PerfMetricsPlugin(RuntimePlugin):
    def __init__(self, include_cuda_memory: bool = True) -> None:
        super().__init__(
            id=PluginId.PROFILER,
            name="perf_metrics",
            runs_after={PluginId.PRECISION, PluginId.GRAD_CLIP},
        )
        self.include_cuda_memory = include_cuda_memory
        self._microbatch_start: float | None = None
        self._forward_start: float | None = None
        self._backward_start: float | None = None
        self._optimizer_start: float | None = None
        self._step_start: float | None = None
        self._metrics: dict[str, MetricValue] = {}

    def on_phase(self, phase: RuntimePhase) -> None:
        now = time.perf_counter()
        if phase == RuntimePhase.PRE_MICROBATCH:
            assert self.runtime is not None
            if bool(self.runtime.state.metadata.get("accum_start", True)):
                self._step_start = now
            self._microbatch_start = now
            return
        if phase == RuntimePhase.PRE_FORWARD:
            self._forward_start = now
            return
        if phase == RuntimePhase.POST_FORWARD:
            self._record_elapsed("perf/forward_sec", self._forward_start, now)
            return
        if phase == RuntimePhase.PRE_BACKWARD:
            self._backward_start = now
            return
        if phase == RuntimePhase.POST_BACKWARD:
            self._record_elapsed("perf/backward_sec", self._backward_start, now)
            self._record_elapsed("perf/microbatch_sec", self._microbatch_start, now)
            return
        if phase == RuntimePhase.PRE_STEP:
            self._optimizer_start = now
            return
        if phase == RuntimePhase.POST_STEP:
            self._record_elapsed("perf/optimizer_sec", self._optimizer_start, now)
            self._record_elapsed("perf/step_sec", self._step_start, now)
            self._step_start = now
            if self.include_cuda_memory:
                self._metrics.update(_cuda_memory_metrics())

    def collect_metrics(self) -> dict[str, MetricValue]:
        return dict(self._metrics)

    def _record_elapsed(self, key: str, start: float | None, end: float) -> None:
        if start is not None:
            self._metrics[key] = end - start


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
