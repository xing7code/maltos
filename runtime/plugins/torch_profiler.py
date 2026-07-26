from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist

from runtime.plugin import PluginId, RuntimePlugin
from runtime.types import MetricValue, RuntimePhase


class TorchProfilerPlugin(RuntimePlugin):
    """PyTorch profiler trace exporter for CUDA/NCCL/operator timeline analysis."""

    def __init__(
        self,
        *,
        trace_dir: str | Path,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        rank0_only: bool = False,
    ) -> None:
        super().__init__(
            id=PluginId.TORCH_PROFILER,
            name="torch_profiler",
            runs_after={PluginId.FP16},
        )
        if wait < 0:
            raise ValueError(f"wait must be >= 0, got {wait}")
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}")
        if active < 1:
            raise ValueError(f"active must be >= 1, got {active}")
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")
        self.trace_dir = Path(trace_dir)
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.rank0_only = rank0_only
        self._profiler: torch.profiler.profile | None = None
        self._enabled = False
        self._trace_path: Path | None = None
        self._trace_file: Path | None = None
        self._post_step_count = 0

    def bind(self, runtime: "RuntimeCore") -> None:
        super().bind(runtime)
        self._start()

    def on_step_phase(self, phase: RuntimePhase) -> None:
        if phase != RuntimePhase.POST_STEP:
            return
        self._post_step_count += 1
        trace_boundary = self._is_rank0_trace_boundary()
        if trace_boundary:
            # Profiler stop synchronizes rank 0's CUDA work.  Keep every peer
            # at the same optimizer-step boundary until that synchronization
            # completes; otherwise they can launch the next collective while
            # rank 0 is still stopping the trace.
            dist.barrier()
        if self._profiler is not None:
            self._profiler.step()
        if trace_boundary:
            dist.barrier()

    def _is_rank0_trace_boundary(self) -> bool:
        if not self.rank0_only or not dist.is_initialized():
            return False
        cycle_steps = self.wait + self.warmup + self.active
        return (
            cycle_steps > 0
            and self._post_step_count <= self.repeat * cycle_steps
            and self._post_step_count % cycle_steps == 0
        )

    def collect_metrics(self) -> dict[str, MetricValue]:
        return {
            "torch_profiler/enabled": self._enabled,
            "torch_profiler/rank0_only": self.rank0_only,
            "torch_profiler/trace_dir": None if self._trace_path is None else str(self._trace_path),
            "torch_profiler/wait": self.wait,
            "torch_profiler/warmup": self.warmup,
            "torch_profiler/active": self.active,
            "torch_profiler/repeat": self.repeat,
        }

    def close(self) -> None:
        if self._profiler is None:
            return
        profiler = self._profiler
        profiler.__exit__(None, None, None)
        self._profiler = None
        if self._trace_file is not None:
            profiler.export_chrome_trace(str(self._trace_file))

    def _start(self) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.rank0_only and rank != 0:
            self._enabled = False
            return

        trace_path = self.trace_dir / f"rank_{rank:05d}"
        trace_path.mkdir(parents=True, exist_ok=True)
        self._trace_path = trace_path
        self._trace_file = trace_path / "trace.json"
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            acc_events=True,
        )
        self._profiler.__enter__()
        self._enabled = True
