"""Runtime integration for the CP-only ByteScale HDP baseline.

HDP layout types live in ``parallel.hdp_helper``; batch scheduling and
materialization live in ``data.bytescale_hdp``. This plugin installs the
rank-local schedule.
"""
from __future__ import annotations

import time

import torch
import torch.distributed as dist
import torch.nn as nn

from parallel.hdp_helper import (
    BYTESCALE_HDP_SCHEDULE_KEY,
    BYTESCALE_HDP_WAVES_KEY,
    ByteScaleHdpBalancedConfig,
    ByteScaleHdpLocalLayout,
)
from runtime.layers.hdp_attention import HdpBalancedAttentionCore
from runtime.mesh import MeshAxis
from runtime.plugin import ContextParallelizableModule, PluginId, RuntimePlugin
from runtime.step_runners import DefaultStepRunner
from runtime.types import RuntimePhase, SetupPhase
from utils.attention_backend import AttentionBackend


class ByteScaleHdpPlugin(RuntimePlugin):
    """Install and execute the rank-local ordered HDP wave schedule."""

    def __init__(self, *, config: ByteScaleHdpBalancedConfig) -> None:
        super().__init__(
            id=PluginId.HDP,
            name="bytescale_hdp",
            runs_after={PluginId.TP, PluginId.TP_SP, PluginId.DP},
            owns_step_runner=True,
        )
        self.config = config
        self._attention_cores: list[HdpBalancedAttentionCore] = []
        self._metrics: dict[str, float | int] = {}
        self._dummy_wave_count = 0
        self._real_wave_count = 0
        self._dummy_wave_seconds = 0.0
        self._real_wave_seconds = 0.0
        self._dummy_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._real_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    @property
    def hdp_group(self) -> dist.ProcessGroup:
        assert self.runtime is not None
        group = self.runtime.get_group(MeshAxis.DCP)
        if group is None:
            raise ValueError("ByteScaleHdpPlugin requires a DCP process group")
        return group

    def bind(self, runtime) -> None:
        super().bind(runtime)
        if any(plugin.id == PluginId.CP for plugin in runtime.plugins if plugin is not self):
            raise ValueError(
                "ByteScaleHdpPlugin and ContextParallelPlugin are mutually exclusive"
            )
        if runtime.mesh.dp * runtime.mesh.cp <= 1:
            raise ValueError("ByteScaleHdpPlugin requires DCP world size > 1")
        if runtime.mesh.cp != 1:
            raise ValueError(
                "ByteScale HDP baseline requires cp_size=1; HDP replaces fixed CP"
            )
        if runtime.mesh.pp != 1:
            raise ValueError(
                "ByteScale HDP currently implements FCP's CP-only baseline and requires pp_size=1"
            )

    def on_setup_phase(self, phase: SetupPhase, model: nn.Module) -> nn.Module:
        if phase is not SetupPhase.TRANSFORM:
            return model
        if dist.get_world_size(self.hdp_group) <= 1:
            return model
        if not isinstance(model, ContextParallelizableModule):
            raise TypeError("ByteScaleHdpPlugin requires context_parallel_spec()")

        assert self.runtime is not None
        for path in model.context_parallel_spec().attention_paths:
            if self.runtime.is_module_path_omitted(path):
                continue
            module = model.get_submodule(path)
            if not hasattr(module, "attn_core"):
                raise TypeError(f"attention module path={path} has no attn_core")
            backend = getattr(module.attn_core, "attention_backend", AttentionBackend.EAGER)
            core = HdpBalancedAttentionCore(self.hdp_group, attention_backend=backend)
            module.attn_core = core
            self._attention_cores.append(core)

        if not self._attention_cores:
            raise ValueError("ByteScaleHdpPlugin found no attention cores to transform")
        return model

    def on_step_phase(self, phase: RuntimePhase) -> None:
        assert self.runtime is not None
        if phase is RuntimePhase.PRE_STEP_RUNNER:
            if self.runtime.state.step_context.accum_start:
                self._reset_wave_overhead()
            schedules = tuple(_schedule_from_batch(wave) for wave in _waves_from_batch(self.runtime.state.batch))
            for schedule in schedules:
                if schedule.partition_tokens != self.config.partition_tokens:
                    raise ValueError(
                        "ByteScale dataloader/plugin partition_tokens mismatch: "
                        f"loader={schedule.partition_tokens} plugin={self.config.partition_tokens}"
                    )
            self.runtime.state.metadata["tokens"] = schedules[0].global_valid_targets
            self._metrics = _wave_metrics(schedules)
        elif phase is RuntimePhase.PRE_FORWARD:
            schedule = _schedule_from_batch(self.runtime.state.batch)
            for core in self._attention_cores:
                core.set_active_schedule(schedule)

    def collect_metrics(self) -> dict[str, float | int]:
        self._drain_ready_event_timings()
        metrics = dict(self._metrics)
        metrics.update(
            {
                "dummy_wave_count": self._dummy_wave_count,
                "real_wave_count": self._real_wave_count,
                "dummy_wave_sec": self._dummy_wave_seconds,
                "real_wave_sec": self._real_wave_seconds,
                "dummy_wave_sec_sum": self._dummy_wave_seconds,
                "real_wave_sec_sum": self._real_wave_seconds,
            }
        )
        self._metrics = {}
        return metrics

    def build_step_runner(self):
        return _HdpWaveStepRunner(self)

    def _reset_wave_overhead(self) -> None:
        self._dummy_wave_count = self._real_wave_count = 0
        self._dummy_wave_seconds = self._real_wave_seconds = 0.0
        self._dummy_events.clear()
        self._real_events.clear()

    def record_wave_overhead(
        self,
        *,
        is_dummy: bool,
        host_seconds: float,
        cuda_events: tuple[torch.cuda.Event, torch.cuda.Event] | None,
    ) -> None:
        if is_dummy:
            self._dummy_wave_count += 1
            self._dummy_wave_seconds += host_seconds
            if cuda_events is not None:
                self._dummy_events.append(cuda_events)
        else:
            self._real_wave_count += 1
            self._real_wave_seconds += host_seconds
            if cuda_events is not None:
                self._real_events.append(cuda_events)

    def _drain_ready_event_timings(self) -> None:
        """Resolve CUDA events only at the normal optimizer-step metric boundary."""
        for events, attribute in (
            (self._dummy_events, "_dummy_wave_seconds"),
            (self._real_events, "_real_wave_seconds"),
        ):
            for start, end in events:
                end.synchronize()
                setattr(self, attribute, getattr(self, attribute) + start.elapsed_time(end) / 1e3)
            events.clear()


class _HdpWaveStepRunner:
    """Run every global HDP wave so ZeRO/DDP module-hook order stays aligned."""

    def __init__(self, plugin: ByteScaleHdpPlugin) -> None:
        self.plugin = plugin

    def run(self, runtime, batch) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        waves = _waves_from_batch(batch)
        for wave_index, wave in enumerate(waves):
            runtime.state.step_context.set_hdp_wave(
                wave_idx=wave_index,
                wave_count=len(waves),
            )
            runtime.state.batch = wave
            layout = _schedule_from_batch(wave)
            is_dummy = not layout.documents
            start_host = time.perf_counter()
            start_event = end_event = None
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            DefaultStepRunner.run_forward(runtime, wave)
            if not torch.is_tensor(runtime.state.loss):
                raise TypeError("ByteScale HDP requires each wave to produce a Tensor loss")
            losses.append(runtime.state.loss.detach())
            DefaultStepRunner.run_backward(runtime)
            if start_event is not None and end_event is not None:
                end_event.record()
                cuda_events = (start_event, end_event)
                host_seconds = 0.0
            else:
                cuda_events = None
                host_seconds = time.perf_counter() - start_host
            self.plugin.record_wave_overhead(
                is_dummy=is_dummy, host_seconds=host_seconds, cuda_events=cuda_events)
        total_loss = torch.stack(losses).sum()
        runtime.state.loss = total_loss
        runtime.state.metadata["raw_loss_for_metrics"] = total_loss
        return total_loss


def _schedule_from_batch(batch) -> ByteScaleHdpLocalLayout:
    if not isinstance(batch, dict):
        raise TypeError("ByteScale HDP requires a dict batch")
    schedule = batch.get(BYTESCALE_HDP_SCHEDULE_KEY)
    if not isinstance(schedule, ByteScaleHdpLocalLayout):
        raise RuntimeError("ByteScaleHdpDataLoader must attach the local schedule")
    return schedule


def _waves_from_batch(batch) -> tuple[dict, ...]:
    if not isinstance(batch, dict):
        raise TypeError("ByteScale HDP requires a dict batch envelope")
    waves = batch.get(BYTESCALE_HDP_WAVES_KEY)
    if not isinstance(waves, tuple) or not waves or not all(isinstance(wave, dict) for wave in waves):
        raise RuntimeError("ByteScaleHdpDataLoader must attach a non-empty tuple of local waves")
    return waves


def _layout_metrics(layout: ByteScaleHdpLocalLayout) -> dict[str, float | int]:
    return {
        "global_documents": layout.global_document_count,
        "local_documents": len(layout.documents),
        "local_valid_tokens": layout.valid_tokens,
        "local_placement_tokens": layout.placement_tokens,
        "local_padded_slots": layout.padded_slots,
        "local_attention_work": sum(
            document.sequence_length * document.local_length
            for document in layout.documents
        ),
    }


def _wave_metrics(layouts: tuple[ByteScaleHdpLocalLayout, ...]) -> dict[str, float | int]:
    metrics = _layout_metrics(layouts[0])
    metrics["local_documents"] = sum(len(layout.documents) for layout in layouts)
    metrics["local_valid_tokens"] = sum(layout.valid_tokens for layout in layouts)
    metrics["local_placement_tokens"] = sum(layout.placement_tokens for layout in layouts)
    metrics["local_padded_slots"] = sum(layout.padded_slots for layout in layouts)
    metrics["local_attention_work"] = sum(
        document.sequence_length * document.local_length
        for layout in layouts
        for document in layout.documents
    )
    metrics["waves"] = len(layouts)
    return metrics
