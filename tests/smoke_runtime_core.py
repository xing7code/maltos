"""Smoke tests for the RuntimeCore control plane.

These tests intentionally avoid torch.distributed so the declarative runtime
pieces remain easy to validate in a single process.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from models import (
    ActivationCheckpointConfig,
    LlamaConfig,
    LlamaForCausalLM,
    OlmoConfig,
    OlmoForCausalLM,
    TinyMoETransformer,
    TinyTransformer,
)
from attention_backend_utils import resolve_attention_backend
from parallel import ParallelPlan
from parallel.specs import TpSpShardAxis
from runtime import DefaultStepRunner, MeshConfig, ParamRole, PluginId, RuntimeCore, RuntimePhase, SetupPhase
from runtime.buffer_allocator import BufferPolicy, acquire_buffer, clear_buffer_pool, release_buffer
from runtime.mesh import MeshAxis
from runtime.plugins.ddp import DataParallelPlugin
from runtime.plugins.grad_clip import GradClipPlugin
from runtime.plugins.fp16 import Fp16Plugin
from runtime.plugins.torch_profiler import TorchProfilerPlugin
from runtime.plugins.tp import TensorParallelPlugin
from runtime.plugin import RuntimePlugin
from state import save_sharded_checkpoint
from utils.attention_backend import AttentionBackend
from utils.constants import INPUT_IDS_KEY, LABELS_KEY, POSITION_IDS_KEY, SEQUENCE_IDS_KEY


class LossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.proj(batch).pow(2).mean()


class SharedExpertModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.ones(1))
        self.expert = nn.Parameter(torch.ones(1))


class FailingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        _ = self.proj(batch)
        raise RuntimeError("boom")


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

    def on_setup_phase(self, phase: SetupPhase, model: nn.Module) -> nn.Module:
        if phase == SetupPhase.TRANSFORM:
            self.event_log.append(f"{self.name}:transform")
        return model

    def on_step_phase(self, phase: RuntimePhase) -> None:
        self.event_log.append(f"{self.name}:{phase.value}")


class PluginStateEcho(RuntimePlugin):
    def __init__(self) -> None:
        super().__init__(id=PluginId.CHECKPOINT, name="plugin_state_echo")
        self.value = 0

    def export_plugin_state(self) -> dict[str, object]:
        return {"value": self.value}

    def import_plugin_state(self, state: dict[str, object]) -> None:
        value = state.get("value")
        if isinstance(value, int):
            self.value = value


class MetricsPlugin(RuntimePlugin):
    def __init__(self) -> None:
        super().__init__(id=PluginId.METRICS, name="metrics_plugin")

    def collect_metrics(self) -> dict[str, float | int | str | bool | None]:
        return {"tokens_per_sec": 12.5, "enabled": True}


class OptimizerOwnerPlugin(RuntimePlugin):
    def __init__(self, plugin_id: PluginId) -> None:
        super().__init__(id=plugin_id, name=f"{plugin_id.value}_owner", owns_optimizer=True)
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    def on_setup_phase(self, phase: SetupPhase, model: nn.Module) -> nn.Module:
        if phase != SetupPhase.FINALIZE:
            return model
        assert self.runtime is not None
        self.optimizer = self.runtime.create_optimizer(model.parameters())
        self.scheduler = self.runtime.create_scheduler(self.optimizer)
        return model

    def optimizer_state_source_rank(self, rank_id: int) -> int:
        return rank_id


class StepRunnerOwnerPlugin(RuntimePlugin):
    def __init__(self, plugin_id: PluginId) -> None:
        super().__init__(id=plugin_id, name=f"{plugin_id.value}_runner", owns_step_runner=True)

    def build_step_runner(self):
        return DefaultStepRunner()


class StrayStepRunnerPlugin(RuntimePlugin):
    def __init__(self, plugin_id: PluginId) -> None:
        super().__init__(id=plugin_id, name=f"{plugin_id.value}_stray_runner")

    def build_step_runner(self):
        return DefaultStepRunner()


class ReplaceLinearPlugin(RuntimePlugin):
    def __init__(self) -> None:
        super().__init__(id=PluginId.CHECKPOINT, name="replace_linear")

    def on_setup_phase(self, phase: SetupPhase, model: nn.Module) -> nn.Module:
        if phase != SetupPhase.TRANSFORM:
            return model
        model.proj = nn.Linear(4, 1)
        return model


def _sgd_factory(lr: float = 0.01):
    return lambda params: torch.optim.SGD(params, lr=lr)


def _adamw_factory(lr: float = 1e-3):
    return lambda params: torch.optim.AdamW(params, lr=lr)


def test_plugin_ordering() -> None:
    events: list[str] = []
    plugins = [
        RecordingPlugin(PluginId.METRICS, events, name="custom_metrics", runs_after={PluginId.CP, PluginId.TP}),
        RecordingPlugin(PluginId.TP, events, name="custom_tensor"),
        RecordingPlugin(PluginId.CHECKPOINT, events, runs_before={PluginId.METRICS}),
    ]
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=LossModel(),
        optimizer_factory=_sgd_factory(),
        plugins=plugins,
    )

    assert [plugin.id for plugin in core.plugins] == [PluginId.TP, PluginId.CHECKPOINT, PluginId.METRICS]
    assert [plugin.name for plugin in core.plugins] == ["custom_tensor", "checkpoint", "custom_metrics"]


def test_runtime_device_is_canonicalized_on_setup() -> None:
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=LossModel(),
        optimizer_factory=_sgd_factory(),
    )
    assert core.device is None
    core.setup()
    assert isinstance(core.device, torch.device)
    assert core.device == torch.device("cpu")


def test_buffer_pool_borrows_views_from_contiguous_slab_and_reuses_returns() -> None:
    clear_buffer_pool()
    device = torch.device("cpu")
    first = acquire_buffer(shape=(8,), dtype=torch.float32, device=device, policy=BufferPolicy.CACHEABLE)
    second = acquire_buffer(shape=(8,), dtype=torch.float32, device=device, policy=BufferPolicy.CACHEABLE)

    assert first.tensor.untyped_storage().data_ptr() == second.tensor.untyped_storage().data_ptr()
    assert first.tensor.data_ptr() != second.tensor.data_ptr()

    first_ptr = first.tensor.data_ptr()
    release_buffer(first)
    release_buffer(second)

    reused = acquire_buffer(shape=(8,), dtype=torch.float32, device=device, policy=BufferPolicy.CACHEABLE)
    assert reused.tensor.data_ptr() == first_ptr
    release_buffer(reused)
    clear_buffer_pool()


def test_buffer_pool_pinned_buffers_share_policy_arena_storage() -> None:
    clear_buffer_pool()
    device = torch.device("cpu")
    first = acquire_buffer(
        shape=(8,),
        dtype=torch.float32,
        device=device,
        policy=BufferPolicy.PINNED,
        key="smoke.pinned.first",
    )
    second = acquire_buffer(
        shape=(8,),
        dtype=torch.float32,
        device=device,
        policy=BufferPolicy.PINNED,
        key="smoke.pinned.second",
    )
    first_again = acquire_buffer(
        shape=(8,),
        dtype=torch.float32,
        device=device,
        policy=BufferPolicy.PINNED,
        key="smoke.pinned.first",
    )

    assert first.tensor.untyped_storage().data_ptr() != second.tensor.untyped_storage().data_ptr()
    assert first_again.tensor.data_ptr() == first.tensor.data_ptr()
    clear_buffer_pool()


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
    optimizer, _ = core.get_optimizer_and_scheduler()
    assert optimizer is not None
    assert {id(param) for group in optimizer.param_groups for param in group["params"]} == new_param_ids


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
    _, should_step = core.run_step(torch.randn(2, 4))
    core.step_optimizer()
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
        plugins=[RecordingPlugin(PluginId.CHECKPOINT, events, name="recorder")],
    )
    core.setup()
    loss, _ = core.run_step(torch.ones(2, 4))
    core.step_optimizer()

    assert loss.ndim == 0
    assert core.state.step == 1
    assert "proj.weight" in core.state_manager.param_states
    assert events == [
        "recorder:transform",
        "recorder:pre_step_runner",
        "recorder:pre_forward",
        "recorder:post_forward",
        "recorder:pre_backward",
        "recorder:post_backward",
        "recorder:pre_step",
        "recorder:post_step",
    ]


def test_multiple_optimizer_plugin_owners_fail() -> None:
    try:
        RuntimeCore(
            mesh=MeshConfig(),
            plan=ParallelPlan(),
            model=LossModel(),
            optimizer_factory=_sgd_factory(),
            plugins=[
                OptimizerOwnerPlugin(PluginId.ZERO1),
                OptimizerOwnerPlugin(PluginId.ZERO2),
            ],
        )
    except ValueError as exc:
        assert "only one optimizer-owning plugin" in str(exc)
    else:
        raise AssertionError("RuntimeCore accepted multiple optimizer-owning plugins")


def test_multiple_step_runner_plugin_owners_fail() -> None:
    try:
        RuntimeCore(
            mesh=MeshConfig(),
            plan=ParallelPlan(),
            model=LossModel(),
            optimizer_factory=_sgd_factory(),
            plugins=[
                StepRunnerOwnerPlugin(PluginId.PP),
                StepRunnerOwnerPlugin(PluginId.CP),
            ],
        )
    except ValueError as exc:
        assert "only one step-runner-owning plugin" in str(exc)
    else:
        raise AssertionError("RuntimeCore accepted multiple step-runner-owning plugins")


def test_stray_step_runner_plugin_fails() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        core = RuntimeCore(
            mesh=MeshConfig(),
            plan=ParallelPlan(),
            model=LossModel(),
            optimizer_factory=_sgd_factory(),
            plugins=[StrayStepRunnerPlugin(PluginId.CP)],
        )
        core.setup()

    assert isinstance(core._step_runner, DefaultStepRunner)
    assert len(caught) == 1
    assert "without declaring owns_step_runner=True" in str(caught[0].message)


def test_runtime_core_requires_optimizer_owner() -> None:
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=LossModel(),
    )
    try:
        core.setup()
    except ValueError as exc:
        assert "optimizer_factory is required" in str(exc)
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
    assert dp_core.optimizer_checkpoint_rank(1) == 0

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
    assert tp_core.optimizer_checkpoint_rank(0) == 0
    assert tp_core.optimizer_checkpoint_rank(1) == 1

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
    assert tp_dp_core.optimizer_checkpoint_rank(2) == 0
    assert tp_dp_core.optimizer_checkpoint_rank(3) == 1

    mixed_model = SharedExpertModel()
    mixed_core = RuntimeCore(
        mesh=MeshConfig(dp=2, tp=1, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=mixed_model,
        optimizer_factory=_sgd_factory(),
        plugins=[DataParallelPlugin()],
    )
    mixed_core.setup()
    mixed_core.set_param_role(mixed_core.model.expert, ParamRole.EXPERT)
    assert mixed_core.should_save_optimizer(0)
    assert mixed_core.should_save_optimizer(1)
    assert mixed_core.optimizer_checkpoint_rank(0) == 0
    assert mixed_core.optimizer_checkpoint_rank(1) == 1


def test_runtime_param_checkpoint_policy() -> None:
    core = RuntimeCore(
        mesh=MeshConfig(dp=2, tp=2, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=SharedExpertModel(),
        optimizer_factory=_sgd_factory(),
    )
    core.setup()
    core.state_manager.update_model_state("shared", replicated_axes={MeshAxis.DP, MeshAxis.TP})
    core.state_manager.update_model_state("expert", replicated_axes={MeshAxis.TP})
    core._resolve_param_checkpoint_metadata()

    assert core.model_state_checkpoint_rank("shared", rank_id=3) == 0
    assert core.model_state_checkpoint_rank("expert", rank_id=3) == 2
    assert core.state_manager.param_states["shared"].source_rank == 0
    assert core.state_manager.param_states["expert"].source_rank == 0


def test_tp_bias_layout_metadata() -> None:
    plugin = TensorParallelPlugin()

    row_linear = nn.Linear(8, 4, bias=True)
    plugin._record_linear_rule("row", row_linear, TpSpShardAxis.PARAM_IN)
    assert plugin._param_shard_axis["row.weight"] == TpSpShardAxis.PARAM_IN
    assert "row.bias" not in plugin._param_shard_axis
    assert "row.bias" in plugin._tp_replicated_params

    col_linear = nn.Linear(4, 8, bias=True)
    plugin._record_linear_rule("col", col_linear, TpSpShardAxis.PARAM_OUT)
    assert plugin._param_shard_axis["col.weight"] == TpSpShardAxis.PARAM_OUT
    assert plugin._param_shard_axis["col.bias"] == TpSpShardAxis.PARAM_OUT


def test_state_manager_exports_only_owned_model_params() -> None:
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=SharedExpertModel(),
        optimizer_factory=_sgd_factory(),
    )
    core.setup()
    core.state_manager.param_states["shared"].source_rank = 0
    core.state_manager.param_states["expert"].source_rank = 1

    state, entries = core.state_manager.export_model_state()

    assert set(state) == {"shared"}
    assert [entry.state_key for entry in entries] == ["shared", "expert"]
    assert [entry.source_rank for entry in entries] == [0, 1]


def test_precision_plugin_metrics_and_clip() -> None:
    model = LossModel()
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        optimizer_factory=_sgd_factory(),
        plugins=[Fp16Plugin(), GradClipPlugin(max_norm=1.0)],
    )
    core.setup()
    _, _ = core.run_step(torch.ones(2, 4))
    core.step_optimizer()
    assert "grad_norm" in core.state.metadata
    assert isinstance(core.state.metadata["overflow"], bool)
    assert core.state.metadata["overflow"] is False
    assert core.state.metadata["loss_scale"] is None
    metrics = core.collect_metrics()
    assert metrics["step"] == 1
    assert metrics["loss"] == float(core.state.loss.detach().float().item())
    assert metrics["lr"] == 0.01
    assert metrics["fp16/loss_scale"] is None
    assert metrics["fp16/overflow"] is False
    assert "grad_clip/grad_norm" in metrics
    assert metrics["grad_clip/max_norm"] == 1.0


def test_precision_plugin_bf16_uses_fp32_master_weights() -> None:
    torch.manual_seed(0)
    model = LossModel()
    bf16_batch = torch.ones(2, 4, dtype=torch.bfloat16)
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        dtype=torch.bfloat16,
        optimizer_factory=_adamw_factory(),
        plugins=[GradClipPlugin(max_norm=1.0)],
    )
    core.setup()

    assert model.proj.weight.dtype == torch.bfloat16
    assert model.proj.bias.dtype == torch.bfloat16

    optimizer, _ = core.get_optimizer_and_scheduler()
    assert optimizer is not None
    master_params = [param for group in optimizer.param_groups for param in group["params"]]
    assert master_params
    assert all(param.dtype == torch.float32 for param in master_params)

    _, should_step = core.run_step(bf16_batch)
    assert should_step is True
    core.step_optimizer()

    state_dict = optimizer.state_dict()
    assert "master_params" in state_dict
    assert all(tensor.dtype == torch.float32 for tensor in state_dict["master_params"])
    inner_state = state_dict["inner_optimizer"]["state"]
    assert inner_state
    adam_state = next(iter(inner_state.values()))
    assert adam_state["exp_avg"].dtype == torch.float32
    assert adam_state["exp_avg_sq"].dtype == torch.float32
    assert model.proj.weight.dtype == torch.bfloat16


def test_precision_plugin_bf16_optimizer_state_roundtrip() -> None:
    torch.manual_seed(0)
    bf16_batch = torch.ones(2, 4, dtype=torch.bfloat16)
    source = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=LossModel(),
        dtype=torch.bfloat16,
        optimizer_factory=_adamw_factory(),
    )
    source.setup()
    _, should_step = source.run_step(bf16_batch)
    assert should_step is True
    source.step_optimizer()
    model_state, _ = source.state_manager.export_model_state()
    optimizer_state = source.state_manager.export_optimizer_state()
    assert optimizer_state is not None

    torch.manual_seed(1)
    target = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=LossModel(),
        dtype=torch.bfloat16,
        optimizer_factory=_adamw_factory(),
    )
    target.setup()
    target.state_manager.import_model_state(model_state)
    target.state_manager.import_optimizer_state(optimizer_state)

    source_optimizer, _ = source.get_optimizer_and_scheduler()
    target_optimizer, _ = target.get_optimizer_and_scheduler()
    assert source_optimizer is not None
    assert target_optimizer is not None

    assert torch.allclose(source.model.proj.weight.float(), target.model.proj.weight.float())
    assert torch.allclose(source.model.proj.bias.float(), target.model.proj.bias.float())

    source_state = source_optimizer.state_dict()
    target_state = target_optimizer.state_dict()
    assert len(source_state["master_params"]) == len(target_state["master_params"])
    for source_tensor, target_tensor in zip(
        source_state["master_params"], target_state["master_params"], strict=True
    ):
        assert torch.allclose(source_tensor, target_tensor)


def test_post_forward_runs_when_model_forward_raises() -> None:
    events: list[str] = []
    plugin = RecordingPlugin(PluginId.METRICS, events, name="recorder")
    bf16_batch = torch.ones(2, 4, dtype=torch.bfloat16)
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=FailingModel(),
        dtype=torch.bfloat16,
        optimizer_factory=_sgd_factory(),
        plugins=[plugin, Fp16Plugin()],
    )
    core.setup()
    try:
        core.run_step(bf16_batch)
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("FailingModel forward should raise")
    assert "recorder:pre_forward" in events
    assert "recorder:post_forward" in events


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
    _, should_step = core.run_step(torch.ones(2, 4))
    metrics = core.collect_metrics()
    assert metrics["metrics/tokens_per_sec"] == 12.5
    assert metrics["metrics/enabled"] is True


def test_torch_profiler_plugin_writes_trace() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model = LossModel()
        core = RuntimeCore(
            mesh=MeshConfig(),
            plan=ParallelPlan(),
            model=model,
            optimizer_factory=_sgd_factory(),
            plugins=[
                TorchProfilerPlugin(
                    trace_dir=tmp,
                    wait=0,
                    warmup=1,
                    active=1,
                    repeat=1,
                )
            ],
        )
        core.setup()
        _, should_step = core.run_step(torch.ones(2, 4))
        core.step_optimizer()
        _, should_step = core.run_step(torch.ones(2, 4))
        core.step_optimizer()
        metrics = core.collect_metrics()
        core.close()

        assert metrics["torch_profiler/enabled"] is True
        assert metrics["torch_profiler/trace_dir"].endswith("rank_00000")
        trace_dir = Path(tmp) / "rank_00000"
        assert trace_dir.is_dir()
        assert any(trace_dir.iterdir())


def test_precision_plugin_fp16_requires_cuda() -> None:
    model = LossModel()
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        dtype=torch.float16,
        optimizer_factory=_sgd_factory(),
        plugins=[Fp16Plugin()],
    )
    try:
        core.setup()
    except ValueError as exc:
        assert "requires CUDA model parameters" in str(exc)
    else:
        raise AssertionError("Fp16Plugin should fail on CPU model parameters")


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
    core.state.step_context.microbatch_idx = 1
    trainer_state = core.state_manager.export_trainer_state()
    assert trainer_state.plugin_states is not None
    assert trainer_state.plugin_states["checkpoint"]["value"] == 7
    assert trainer_state.step_context["microbatch_idx"] == 1
    echo.value = 0
    core.state.step_context.microbatch_idx = 0
    core.state_manager.import_trainer_state(trainer_state)
    assert echo.value == 7
    assert core.state.step_context.microbatch_idx == 1


def test_runtime_setup_weights_only_checkpoint_load_skips_runtime_state() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp) / "weights_only_ckpt"
        source_model = LossModel()
        source_plugin = PluginStateEcho()
        source_core = RuntimeCore(
            mesh=MeshConfig(),
            plan=ParallelPlan(),
            model=source_model,
            optimizer_factory=_adamw_factory(),
            plugins=[source_plugin],
        )
        source_core.setup()
        source_batch = torch.ones(2, 4)
        _, should_step = source_core.run_step(source_batch)
        assert should_step is True
        source_core.step_optimizer()
        source_plugin.value = 9
        source_weight = source_model.proj.weight.detach().clone()
        save_sharded_checkpoint(
            source_core.state_manager,
            checkpoint_dir,
        )
        source_core.close()

        restored_model = LossModel()
        restored_plugin = PluginStateEcho()
        restored_core = RuntimeCore(
            mesh=MeshConfig(),
            plan=ParallelPlan(),
            model=restored_model,
            optimizer_factory=_adamw_factory(),
            plugins=[restored_plugin],
        )
        restored_core.setup(checkpoint_path=checkpoint_dir, load_weights_only=True)

        optimizer, _ = restored_core.get_optimizer_and_scheduler()
        assert optimizer is not None
        assert torch.allclose(restored_model.proj.weight.detach(), source_weight)
        assert restored_plugin.value == 0
        assert restored_core.state.step_context.step == 0
        assert optimizer.state_dict()["state"] == {}
        restored_core.close()


def test_grad_accumulation_runtime_step_cadence() -> None:
    model = LossModel()
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        grad_accum_steps=2,
        optimizer_factory=_sgd_factory(),
    )
    core.setup()
    w0 = model.proj.weight.detach().clone()
    _, should_step = core.run_step(torch.ones(2, 4))
    w1 = model.proj.weight.detach().clone()
    assert core.state.step == 0
    assert torch.allclose(w0, w1)
    assert should_step is False
    assert core.state.step_context is not None
    assert core.state.step_context.microbatch_idx == 1
    _, should_step = core.run_step(torch.ones(2, 4))
    assert should_step is True
    core.step_optimizer()
    w2 = model.proj.weight.detach().clone()
    assert core.state.step == 1
    assert not torch.allclose(w1, w2)
    assert core.state.step_context is not None
    assert core.state.step_context.microbatch_idx == 0


def test_grad_accumulation_resume_boundary_cadence() -> None:
    model = LossModel()
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        grad_accum_steps=2,
        optimizer_factory=_sgd_factory(),
    )
    core.setup()
    _, should_step = core.run_step(torch.ones(2, 4))
    trainer_state = core.state_manager.export_trainer_state()
    resumed_model = LossModel()
    resumed = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=resumed_model,
        grad_accum_steps=2,
        optimizer_factory=_sgd_factory(),
    )
    resumed.setup()
    resumed.state_manager.import_trainer_state(trainer_state)
    _, should_step = resumed.run_step(torch.ones(2, 4))
    assert should_step is True
    assert resumed.state.step_context is not None
    assert resumed.state.step_context.microbatch_idx == 0
    resumed.step_optimizer()
    assert resumed.state.step == 1
    _, should_step = resumed.run_step(torch.ones(2, 4))
    assert should_step is False
    assert resumed.state.step_context is not None
    assert resumed.state.step_context.microbatch_idx == 1
    assert resumed.state.step == 1


def test_step_optimizer_requires_optimizer() -> None:
    core = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=LossModel(),
    )
    core.setup()
    try:
        core.step_optimizer()
    except RuntimeError as exc:
        assert "requires a runtime-owned or plugin-owned optimizer" in str(exc)
    else:
        raise AssertionError("expected step_optimizer() to fail without an optimizer")


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
        INPUT_IDS_KEY: torch.randint(0, 32, (2, 8)),
        LABELS_KEY: torch.randint(0, 32, (2, 8)),
    }

    core.setup()
    before = {name: param.detach().clone() for name, param in core.model.named_parameters()}
    loss, _ = core.run_step(batch)
    core.step_optimizer()

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
    sdpa = LlamaForCausalLM(LlamaConfig(**base_config, attention_backend=AttentionBackend.SDPA_AUTO))
    eager = LlamaForCausalLM(LlamaConfig(**base_config, attention_backend=AttentionBackend.EAGER))
    eager.load_state_dict(sdpa.state_dict())
    sdpa.eval()
    eager.eval()
    input_ids = torch.randint(0, 32, (2, 8))

    with torch.no_grad():
        sdpa_logits = sdpa(input_ids)
        eager_logits = eager(input_ids)

    torch.testing.assert_close(sdpa_logits, eager_logits, atol=1e-5, rtol=1e-5)


def test_llama_flash_attn_backend_matches_eager_attention_for_packed_batch() -> None:
    torch.manual_seed(4321)
    attention_backend = resolve_attention_backend("auto")
    base_config = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=8,
    )
    flash = LlamaForCausalLM(LlamaConfig(**base_config, attention_backend=attention_backend))
    eager = LlamaForCausalLM(LlamaConfig(**base_config, attention_backend=AttentionBackend.EAGER))
    eager.load_state_dict(flash.state_dict())
    flash.eval()
    eager.eval()
    batch = {
        INPUT_IDS_KEY: torch.randint(0, 32, (2, 8)),
        POSITION_IDS_KEY: torch.tensor(
            [
                [0, 1, 2, 3, 0, 1, 2, 3],
                [0, 1, 2, 3, 0, 1, 2, 3],
            ],
            dtype=torch.long,
        ),
        SEQUENCE_IDS_KEY: torch.tensor(
            [
                [0, 0, 0, 0, 1, 1, 1, 1],
                [2, 2, 2, 2, 3, 3, 3, 3],
            ],
            dtype=torch.long,
        ),
    }

    with torch.no_grad():
        flash_logits = flash(batch)
        eager_logits = eager(batch)

    torch.testing.assert_close(flash_logits, eager_logits, atol=1e-4, rtol=1e-4)


def test_olmo_activation_checkpointing_train_step() -> None:
    torch.manual_seed(9876)
    model = OlmoForCausalLM(
        OlmoConfig(
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
        INPUT_IDS_KEY: torch.randint(0, 32, (2, 8)),
        LABELS_KEY: torch.randint(0, 32, (2, 8)),
    }

    core.setup()
    before = {name: param.detach().clone() for name, param in core.model.named_parameters()}
    loss, _ = core.run_step(batch)
    core.step_optimizer()

    assert loss.ndim == 0
    assert core.state.step == 1
    assert any(
        not torch.equal(before[name], param.detach())
        for name, param in core.model.named_parameters()
        if param.requires_grad
    )


def test_olmo_flash_attn_backend_matches_eager_attention_for_packed_batch() -> None:
    torch.manual_seed(6543)
    attention_backend = resolve_attention_backend("auto")
    base_config = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=8,
    )
    flash = OlmoForCausalLM(OlmoConfig(**base_config, attention_backend=attention_backend))
    eager = OlmoForCausalLM(OlmoConfig(**base_config, attention_backend=AttentionBackend.EAGER))
    eager.load_state_dict(flash.state_dict())
    flash.eval()
    eager.eval()
    batch = _packed_test_batch()

    with torch.no_grad():
        flash_logits = flash(batch)
        eager_logits = eager(batch)

    torch.testing.assert_close(flash_logits, eager_logits, atol=1e-4, rtol=1e-4)


def _packed_test_batch() -> dict[str, torch.Tensor]:
    return {
        INPUT_IDS_KEY: torch.randint(0, 32, (2, 8)),
        POSITION_IDS_KEY: torch.tensor(
            [
                [0, 1, 2, 3, 0, 1, 2, 3],
                [0, 1, 2, 3, 0, 1, 2, 3],
            ],
            dtype=torch.long,
        ),
        SEQUENCE_IDS_KEY: torch.tensor(
            [
                [0, 0, 0, 0, 1, 1, 1, 1],
                [2, 2, 2, 2, 3, 3, 3, 3],
            ],
            dtype=torch.long,
        ),
    }


def test_tiny_flash_attn_backend_matches_eager_attention_for_packed_batch() -> None:
    torch.manual_seed(2468)
    attention_backend = resolve_attention_backend("auto")
    common_kwargs = dict(
        dim=16,
        n_heads=4,
        n_kv_heads=4,
        hidden_size=32,
        eps=1e-5,
        n_layers=2,
        vocab_size=32,
        max_seq_len=8,
    )
    flash = TinyTransformer(**common_kwargs, attention_backend=attention_backend)
    eager = TinyTransformer(**common_kwargs, attention_backend=AttentionBackend.EAGER)
    eager.load_state_dict(flash.state_dict())
    flash.eval()
    eager.eval()
    batch = _packed_test_batch()

    with torch.no_grad():
        flash_logits = flash(batch)
        eager_logits = eager(batch)

    torch.testing.assert_close(flash_logits, eager_logits, atol=1e-4, rtol=1e-4)


def test_tiny_moe_flash_attn_backend_matches_eager_attention_for_packed_batch() -> None:
    torch.manual_seed(8642)
    attention_backend = resolve_attention_backend("auto")
    common_kwargs = dict(
        dim=16,
        n_heads=4,
        n_kv_heads=4,
        hidden_size=32,
        eps=1e-5,
        n_layers=2,
        vocab_size=32,
        max_seq_len=8,
        num_experts=4,
    )
    flash = TinyMoETransformer(**common_kwargs, attention_backend=attention_backend)
    eager = TinyMoETransformer(**common_kwargs, attention_backend=AttentionBackend.EAGER)
    eager.load_state_dict(flash.state_dict())
    flash.eval()
    eager.eval()
    batch = _packed_test_batch()

    with torch.no_grad():
        flash_logits = flash(batch)
        eager_logits = eager(batch)

    torch.testing.assert_close(flash_logits, eager_logits, atol=1e-4, rtol=1e-4)


def main() -> None:
    test_plugin_ordering()
    test_buffer_pool_borrows_views_from_contiguous_slab_and_reuses_returns()
    test_buffer_pool_pinned_buffers_share_policy_arena_storage()
    test_optimizer_factory_runs_after_model_transform()
    test_scheduler_factory_supports_plugin_owned_optimizer()
    test_missing_required_plugin_fails()
    test_train_step_phases_and_state_manager()
    test_multiple_optimizer_plugin_owners_fail()
    test_runtime_core_requires_optimizer_owner()
    test_runtime_optimizer_checkpoint_policy()
    test_runtime_param_checkpoint_policy()
    test_tp_bias_layout_metadata()
    test_state_manager_exports_only_owned_model_params()
    test_precision_plugin_metrics_and_clip()
    test_post_forward_runs_when_model_forward_raises()
    test_runtime_collects_plugin_metrics()
    test_torch_profiler_plugin_writes_trace()
    test_precision_plugin_fp16_requires_cuda()
    test_trainer_state_plugin_states_roundtrip()
    test_grad_accumulation_runtime_step_cadence()
    test_grad_accumulation_resume_boundary_cadence()
    test_llama_activation_checkpointing_train_step()
    test_llama_sdpa_auto_matches_eager_attention()
    test_llama_flash_attn_backend_matches_eager_attention_for_packed_batch()
    test_olmo_activation_checkpointing_train_step()
    test_olmo_flash_attn_backend_matches_eager_attention_for_packed_batch()
    test_tiny_flash_attn_backend_matches_eager_attention_for_packed_batch()
    test_tiny_moe_flash_attn_backend_matches_eager_attention_for_packed_batch()
    print("runtime core smoke ok")


if __name__ == "__main__":
    main()
