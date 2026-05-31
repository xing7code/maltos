"""Smoke tests for the RuntimeCore control plane.

These tests intentionally avoid torch.distributed so the declarative runtime
pieces remain easy to validate in a single process.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models import ActivationCheckpointConfig, LlamaConfig, LlamaForCausalLM
from parallel import ParallelPlan
from runtime import MeshConfig, PluginId, RuntimeCore, RuntimePhase
from runtime.plugins.ddp import DataParallelPlugin
from runtime.plugins.grad_clip import GradClipPlugin
from runtime.plugins.precision import PrecisionPlugin
from runtime.plugins.tp import TensorParallelPlugin
from runtime.plugin import RuntimePlugin


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


class PluginStateEcho(RuntimePlugin):
    def __init__(self) -> None:
        super().__init__(id=PluginId.PROFILER, name="plugin_state_echo")
        self.value = 0

    def export_plugin_state(self) -> dict[str, object]:
        return {"value": self.value}

    def import_plugin_state(self, state: dict[str, object]) -> None:
        value = state.get("value")
        if isinstance(value, int):
            self.value = value


class MetricsPlugin(RuntimePlugin):
    def __init__(self) -> None:
        super().__init__(id=PluginId.PROFILER, name="metrics_plugin")

    def collect_metrics(self) -> dict[str, float | int | str | bool | None]:
        return {"tokens_per_sec": 12.5, "enabled": True}


class OptimizerOwnerPlugin(RuntimePlugin):
    def __init__(self, plugin_id: PluginId) -> None:
        super().__init__(id=plugin_id, name=f"{plugin_id.value}_owner", owns_optimizer=True)
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    def transform_model(self, model: nn.Module) -> nn.Module:
        assert self.runtime is not None
        self.optimizer = self.runtime.create_optimizer(model.parameters())
        self.scheduler = self.runtime.create_scheduler(self.optimizer)
        return model


class ReplaceLinearPlugin(RuntimePlugin):
    def __init__(self) -> None:
        super().__init__(id=PluginId.PROFILER, name="replace_linear")

    def transform_model(self, model: nn.Module) -> nn.Module:
        model.proj = nn.Linear(4, 1)
        return model


def _sgd_factory(lr: float = 0.01):
    return lambda params: torch.optim.SGD(params, lr=lr)


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
        model=LossModel(),
        optimizer_factory=_sgd_factory(),
        plugins=plugins,
    )

    assert [plugin.id for plugin in core.plugins] == [PluginId.TP, PluginId.CHECKPOINT, PluginId.PROFILER]
    assert [plugin.name for plugin in core.plugins] == ["custom_tensor", "checkpoint", "custom_profiler"]


def test_optimizer_factory_runs_after_model_transform() -> None:
    captured_param_ids: list[int] = []

    def build_optimizer(params) -> torch.optim.Optimizer:
        params = list(params)
        captured_param_ids.extend(id(param) for param in params)
        return torch.optim.SGD(params, lr=0.01)

    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=LossModel(),
        optimizer_factory=build_optimizer,
        plugins=[ReplaceLinearPlugin()],
    )
    old_param_ids = {id(param) for param in core.model.parameters()}
    core.setup()
    new_param_ids = {id(param) for param in core.model.parameters()}

    assert old_param_ids.isdisjoint(new_param_ids)
    assert set(captured_param_ids) == new_param_ids
    assert {id(param) for group in core.optimizer.param_groups for param in group["params"]} == new_param_ids


def test_scheduler_factory_supports_plugin_owned_optimizer() -> None:
    plugin = OptimizerOwnerPlugin(PluginId.ZERO1)
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=LossModel(),
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.01),
        scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
        plugins=[plugin],
    )
    core.setup()
    assert plugin.optimizer is not None
    assert plugin.scheduler is not None
    assert core.get_optimizer_and_scheduler() == (plugin.optimizer, plugin.scheduler)
    core.run_train_step(torch.randn(2, 4))
    assert plugin.scheduler.last_epoch == 1
    assert plugin.optimizer.param_groups[0]["lr"] == 0.005


def test_missing_required_plugin_fails() -> None:
    try:
        RuntimeCore(
            mesh=MeshConfig(),
            plan=ParallelPlan(),
            model=LossModel(),
            optimizer_factory=_sgd_factory(),
            plugins=[RecordingPlugin(PluginId.SP, [], requires={PluginId.TP})],
        )
    except ValueError as exc:
        assert "requires missing plugins=['tp']" in str(exc)
    else:
        raise AssertionError("RuntimeCore accepted a plugin with a missing hard dependency")


def test_train_step_phases_and_state_manager() -> None:
    events: list[str] = []
    model = LossModel()
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        optimizer_factory=_sgd_factory(),
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


def test_multiple_optimizer_plugin_owners_fail() -> None:
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=LossModel(),
        plugins=[
            OptimizerOwnerPlugin(PluginId.ZERO1),
            OptimizerOwnerPlugin(PluginId.ZERO2),
        ],
    )
    try:
        core.setup()
    except ValueError as exc:
        assert "only one optimizer-owning plugin" in str(exc)
    else:
        raise AssertionError("RuntimeCore accepted multiple optimizer-owning plugins")


def test_runtime_core_requires_optimizer_owner() -> None:
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=LossModel(),
    )
    try:
        core.setup()
    except ValueError as exc:
        assert "requires optimizer_factory" in str(exc)
    else:
        raise AssertionError("RuntimeCore accepted a config without an optimizer owner")


def test_runtime_optimizer_checkpoint_policy() -> None:
    dp_model = LossModel()
    dp_core = RuntimeCore(
        mesh=MeshConfig(dp=2, tp=1, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=dp_model,
        optimizer_factory=_sgd_factory(),
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
        optimizer_factory=_sgd_factory(),
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
        optimizer_factory=_sgd_factory(),
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


def test_precision_plugin_metrics_and_clip() -> None:
    model = LossModel()
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        optimizer_factory=_sgd_factory(),
        plugins=[PrecisionPlugin(compute_dtype=None), GradClipPlugin(max_norm=1.0)],
    )
    core.setup()
    _ = core.run_train_step(torch.ones(2, 4))
    assert "grad_norm" in core.state.metadata
    assert isinstance(core.state.metadata["overflow"], bool)
    assert core.state.metadata["overflow"] is False
    assert core.state.metadata["loss_scale"] is None
    metrics = core.collect_metrics()
    assert metrics["step"] == 1
    assert metrics["loss"] == float(core.state.loss.detach().float().item())
    assert metrics["lr"] == 0.01
    assert metrics["precision/loss_scale"] is None
    assert metrics["precision/overflow"] is False
    assert "grad_clip/grad_norm" in metrics
    assert metrics["grad_clip/max_norm"] == 1.0


def test_runtime_collects_plugin_metrics() -> None:
    model = LossModel()
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        optimizer_factory=_sgd_factory(),
        plugins=[MetricsPlugin()],
    )
    core.setup()
    core.run_train_step(torch.ones(2, 4))
    metrics = core.collect_metrics()
    assert metrics["profiler/tokens_per_sec"] == 12.5
    assert metrics["profiler/enabled"] is True


def test_precision_plugin_fp16_requires_cuda() -> None:
    model = LossModel()
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        optimizer_factory=_sgd_factory(),
        plugins=[PrecisionPlugin(compute_dtype=torch.float16)],
    )
    try:
        core.setup()
    except ValueError as exc:
        assert "requires CUDA model parameters" in str(exc)
    else:
        raise AssertionError("PrecisionPlugin(fp16) should fail on CPU model parameters")


def test_trainer_state_plugin_states_roundtrip() -> None:
    model = LossModel()
    echo = PluginStateEcho()
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        optimizer_factory=_sgd_factory(),
        plugins=[echo],
    )
    core.setup()
    echo.value = 7
    core.state.microbatch_idx = 1
    trainer_state = core.state_manager.export_trainer_state()
    assert trainer_state.plugin_states is not None
    assert trainer_state.plugin_states["profiler"]["value"] == 7
    assert trainer_state.microbatch_idx == 1
    echo.value = 0
    core.state.microbatch_idx = 0
    core.state_manager.import_trainer_state(trainer_state)
    assert echo.value == 7
    assert core.state.microbatch_idx == 1


def test_grad_accumulation_runtime_step_cadence() -> None:
    model = LossModel()
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        optimizer_factory=_sgd_factory(),
        grad_accum_steps=2,
    )
    core.setup()
    w0 = model.proj.weight.detach().clone()
    core.run_train_step(torch.ones(2, 4))
    w1 = model.proj.weight.detach().clone()
    assert core.state.step == 0
    assert torch.allclose(w0, w1)
    assert core.state.metadata["should_step_optimizer"] is False
    core.run_train_step(torch.ones(2, 4))
    w2 = model.proj.weight.detach().clone()
    assert core.state.step == 1
    assert not torch.allclose(w1, w2)
    assert core.state.metadata["should_step_optimizer"] is True


def test_grad_accumulation_resume_boundary_cadence() -> None:
    model = LossModel()
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        optimizer_factory=_sgd_factory(),
        grad_accum_steps=2,
    )
    core.setup()
    core.run_train_step(torch.ones(2, 4))
    trainer_state = core.state_manager.export_trainer_state()
    resumed_model = LossModel()
    resumed = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=resumed_model,
        optimizer_factory=_sgd_factory(),
        grad_accum_steps=2,
    )
    resumed.setup()
    resumed.state_manager.import_trainer_state(trainer_state)
    resumed.run_train_step(torch.ones(2, 4))
    assert resumed.state.metadata["should_step_optimizer"] is True
    assert resumed.state.step == 1
    resumed.run_train_step(torch.ones(2, 4))
    assert resumed.state.metadata["should_step_optimizer"] is False
    assert resumed.state.step == 1


def test_llama_activation_checkpointing_train_step() -> None:
    torch.manual_seed(1234)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=8,
            activation_checkpointing=ActivationCheckpointConfig(enabled=True, every_n_layers=2),
        )
    )
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        optimizer_factory=_sgd_factory(),
    )
    batch = {
        "input_ids": torch.randint(0, 32, (2, 8)),
        "labels": torch.randint(0, 32, (2, 8)),
    }

    core.setup()
    before = {name: param.detach().clone() for name, param in core.model.named_parameters()}
    loss = core.run_train_step(batch)

    assert loss.ndim == 0
    assert core.state.step == 1
    assert any(
        not torch.equal(before[name], param.detach())
        for name, param in core.model.named_parameters()
        if param.requires_grad
    )


def test_llama_sdpa_auto_matches_eager_attention() -> None:
    torch.manual_seed(1234)
    base_config = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=8,
    )
    sdpa = LlamaForCausalLM(LlamaConfig(**base_config, attention_backend="sdpa_auto"))
    eager = LlamaForCausalLM(LlamaConfig(**base_config, attention_backend="eager"))
    eager.load_state_dict(sdpa.state_dict())
    sdpa.eval()
    eager.eval()
    input_ids = torch.randint(0, 32, (2, 8))

    with torch.no_grad():
        sdpa_logits = sdpa(input_ids)
        eager_logits = eager(input_ids)

    torch.testing.assert_close(sdpa_logits, eager_logits, atol=1e-5, rtol=1e-5)


def main() -> None:
    test_plugin_ordering()
    test_optimizer_factory_runs_after_model_transform()
    test_scheduler_factory_supports_plugin_owned_optimizer()
    test_missing_required_plugin_fails()
    test_train_step_phases_and_state_manager()
    test_multiple_optimizer_plugin_owners_fail()
    test_runtime_core_requires_optimizer_owner()
    test_runtime_optimizer_checkpoint_policy()
    test_precision_plugin_metrics_and_clip()
    test_runtime_collects_plugin_metrics()
    test_precision_plugin_fp16_requires_cuda()
    test_trainer_state_plugin_states_roundtrip()
    test_grad_accumulation_runtime_step_cadence()
    test_grad_accumulation_resume_boundary_cadence()
    test_llama_activation_checkpointing_train_step()
    test_llama_sdpa_auto_matches_eager_attention()
    print("runtime core smoke ok")


if __name__ == "__main__":
    main()
