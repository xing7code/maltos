"""Smoke tests for the RuntimeCore control plane.

These tests intentionally avoid torch.distributed so the declarative runtime
pieces remain easy to validate in a single process.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from train_system.parallel import ParallelPlan
from train_system.runtime import MeshConfig, PluginId, RuntimeCore, RuntimePhase
from train_system.runtime.plugins.ddp import DataParallelPlugin
from train_system.runtime.plugins.tp import TensorParallelPlugin
from train_system.runtime.plugin import RuntimePlugin
from train_system.runtime.plugins.zero1 import Zero1Plugin
from train_system.runtime.plugins.zero2 import Zero2Plugin


class LossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.proj(batch).pow(2).mean()


class RecordingPlugin(RuntimePlugin):
    def __init__(
        self,
        plugin_id: PluginId,
        event_log: list[str],
        name: str | None = None,
        requires: set[PluginId] | None = None,
        runs_after: set[PluginId] | None = None,
        runs_before: set[PluginId] | None = None,
    ) -> None:
        super().__init__(
            id=plugin_id,
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
        RecordingPlugin(PluginId.PROFILER, events, name="custom_profiler", runs_after={PluginId.CP, PluginId.TP}),
        RecordingPlugin(PluginId.TP, events, name="custom_tensor"),
        RecordingPlugin(PluginId.CHECKPOINT, events, runs_before={PluginId.PROFILER}),
    ]
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=(model := LossModel()),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        plugins=plugins,
    )

    assert [plugin.id for plugin in core.plugins] == [PluginId.TP, PluginId.CHECKPOINT, PluginId.PROFILER]
    assert [plugin.name for plugin in core.plugins] == ["custom_tensor", "checkpoint", "custom_profiler"]


def test_missing_required_plugin_fails() -> None:
    try:
        RuntimeCore(
            mesh=MeshConfig(),
            plan=ParallelPlan(),
            model=(model := LossModel()),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            plugins=[RecordingPlugin(PluginId.SP, [], requires={PluginId.TP})],
        )
    except ValueError as exc:
        assert "requires missing plugins=['tp']" in str(exc)
    else:
        raise AssertionError("RuntimeCore accepted a plugin with a missing hard dependency")


def test_train_step_phases_and_state_manager() -> None:
    events: list[str] = []
    model = LossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        optimizer=optimizer,
        plugins=[RecordingPlugin(PluginId.PROFILER, events, name="recorder")],
    )
    core.setup()
    loss = core.run_train_step(torch.ones(2, 4))

    assert loss.ndim == 0
    assert core.state.step == 1
    assert "proj.weight" in core.state_manager.param_states
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


def test_duplicate_optimizer_owners_fail() -> None:
    model = LossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    try:
        RuntimeCore(
            mesh=MeshConfig(dp=2, tp=1, pp=1, cp=1, ep=1),
            plan=ParallelPlan(),
            model=model,
            optimizer=optimizer,
            plugins=[Zero1Plugin(optimizer_cls=torch.optim.SGD, lr=0.01)],
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("RuntimeCore accepted both runtime and plugin-owned optimizers")


def test_multiple_optimizer_plugin_owners_fail() -> None:
    try:
        RuntimeCore(
            mesh=MeshConfig(dp=2, tp=1, pp=1, cp=1, ep=1),
            plan=ParallelPlan(),
            model=LossModel(),
            plugins=[
                Zero1Plugin(optimizer_cls=torch.optim.SGD, lr=0.01),
                Zero2Plugin(optimizer_cls=torch.optim.SGD, lr=0.01),
            ],
        )
    except ValueError as exc:
        assert "only one optimizer-owning plugin" in str(exc)
    else:
        raise AssertionError("RuntimeCore accepted multiple optimizer-owning plugins")


def test_runtime_core_requires_optimizer_owner() -> None:
    try:
        RuntimeCore(
            mesh=MeshConfig(),
            plan=ParallelPlan(),
            model=LossModel(),
        )
    except ValueError as exc:
        assert "exactly one optimizer owner" in str(exc)
    else:
        raise AssertionError("RuntimeCore accepted a config without an optimizer owner")


def test_runtime_optimizer_checkpoint_policy() -> None:
    dp_model = LossModel()
    dp_core = RuntimeCore(
        mesh=MeshConfig(dp=2, tp=1, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=dp_model,
        optimizer=torch.optim.SGD(dp_model.parameters(), lr=0.01),
        plugins=[DataParallelPlugin()],
    )
    for plugin in dp_core.plugins:
        plugin.bind(dp_core)
    assert dp_core.should_save_optimizer(0)
    assert not dp_core.should_save_optimizer(1)
    assert dp_core.optimizer_state_source_rank(1) == 0

    tp_model = LossModel()
    tp_core = RuntimeCore(
        mesh=MeshConfig(dp=1, tp=2, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=tp_model,
        optimizer=torch.optim.SGD(tp_model.parameters(), lr=0.01),
        plugins=[TensorParallelPlugin()],
    )
    for plugin in tp_core.plugins:
        plugin.bind(tp_core)
    assert tp_core.should_save_optimizer(0)
    assert tp_core.should_save_optimizer(1)
    assert tp_core.optimizer_state_source_rank(0) == 0
    assert tp_core.optimizer_state_source_rank(1) == 1

    tp_dp_model = LossModel()
    tp_dp_core = RuntimeCore(
        mesh=MeshConfig(dp=2, tp=2, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=tp_dp_model,
        optimizer=torch.optim.SGD(tp_dp_model.parameters(), lr=0.01),
        plugins=[TensorParallelPlugin(), DataParallelPlugin()],
    )
    for plugin in tp_dp_core.plugins:
        plugin.bind(tp_dp_core)
    assert tp_dp_core.should_save_optimizer(0)
    assert tp_dp_core.should_save_optimizer(1)
    assert not tp_dp_core.should_save_optimizer(2)
    assert not tp_dp_core.should_save_optimizer(3)
    assert tp_dp_core.optimizer_state_source_rank(2) == 0
    assert tp_dp_core.optimizer_state_source_rank(3) == 1


def main() -> None:
    test_plugin_ordering()
    test_missing_required_plugin_fails()
    test_train_step_phases_and_state_manager()
    test_duplicate_optimizer_owners_fail()
    test_multiple_optimizer_plugin_owners_fail()
    test_runtime_core_requires_optimizer_owner()
    test_runtime_optimizer_checkpoint_policy()
    print("runtime core smoke ok")


if __name__ == "__main__":
    main()
