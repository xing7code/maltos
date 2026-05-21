"""Smoke tests for the RuntimeCore control plane.

These tests intentionally avoid torch.distributed so the declarative runtime
pieces remain easy to validate in a single process.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from train_system.parallel import ParallelPlan
from train_system.runtime import MeshConfig, RuntimeCore, RuntimePhase
from train_system.runtime.plugin import RuntimePlugin


class LossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.proj(batch).pow(2).mean()


class RecordingPlugin(RuntimePlugin):
    def __init__(
        self,
        name: str,
        event_log: list[str],
        requires: set[str] | None = None,
        runs_after: set[str] | None = None,
        runs_before: set[str] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            requires=requires or set(),
            runs_after=runs_after or set(),
            runs_before=runs_before or set(),
        )
        self.event_log = event_log

    def transform_model(self, model: nn.Module) -> nn.Module:
        self.event_log.append(f"{self.name}:transform")
        return model

    def on_phase(self, phase: RuntimePhase) -> None:
        self.event_log.append(f"{self.name}:{phase.value}")


def test_plugin_ordering() -> None:
    events: list[str] = []
    plugins = [
        RecordingPlugin("profiler", events, runs_after={"missing_optional", "tp"}),
        RecordingPlugin("tp", events),
        RecordingPlugin("checkpoint", events, runs_before={"profiler"}),
    ]
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=LossModel(),
        plugins=plugins,
    )

    assert [plugin.name for plugin in core.plugins] == ["tp", "checkpoint", "profiler"]


def test_missing_required_plugin_fails() -> None:
    try:
        RuntimeCore(
            mesh=MeshConfig(),
            plan=ParallelPlan(),
            model=LossModel(),
            plugins=[RecordingPlugin("sp", [], requires={"tp"})],
        )
    except ValueError as exc:
        assert "requires missing plugins=['tp']" in str(exc)
    else:
        raise AssertionError("RuntimeCore accepted a plugin with a missing hard dependency")


def test_train_step_phases_and_state_registry() -> None:
    events: list[str] = []
    model = LossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        optimizer=optimizer,
        plugins=[RecordingPlugin("recorder", events)],
    )
    core.setup()
    loss = core.run_train_step(torch.ones(2, 4))

    assert loss.ndim == 0
    assert core.state.step == 1
    assert "proj.weight" in core.state_registry.params
    assert events == [
        "recorder:setup",
        "recorder:transform",
        "recorder:transform_model",
        "recorder:pre_microbatch",
        "recorder:pre_forward",
        "recorder:post_forward",
        "recorder:pre_backward",
        "recorder:post_backward",
        "recorder:pre_step",
        "recorder:post_step",
    ]


def main() -> None:
    test_plugin_ordering()
    test_missing_required_plugin_fails()
    test_train_step_phases_and_state_registry()
    print("runtime core smoke ok")


if __name__ == "__main__":
    main()
