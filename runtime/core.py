from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from pathlib import Path
from typing import TYPE_CHECKING, Any
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist

from parallel.plan import ParallelPlan
from runtime.mesh import MeshAxis, MeshConfig, ProcessGroupManager
from runtime.optim import MasterWeightsOptimizer
from runtime.plugin import (
    FlopsEstimatableModule,
    ModelStateOwner,
    OptimizerOwner,
    PluginId,
    RuntimePlugin,
    StepRunnerOwner,
)
from runtime.step_runners import DefaultStepRunner
from runtime.types import MetricValue, ParamRole, RuntimePhase, RuntimeState, SetupPhase, StepContext
from state.state import ModelStateMeta, StateManager
from state.checkpoint import load_sharded_checkpoint
from utils.constants import INPUT_IDS_KEY

if TYPE_CHECKING:
    from runtime.step_runners import StepRunner

OptimizerFactory = Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]
SchedulerFactory = Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]


@dataclass
class RuntimeCore:
    model: nn.Module
    mesh: MeshConfig = field(default_factory=MeshConfig)
    plan: ParallelPlan = field(default_factory=ParallelPlan)
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    grad_accum_steps: int = 1
    grad_clip_max_norm: float | None = None
    optimizer_factory: OptimizerFactory | None = None
    scheduler_factory: SchedulerFactory | None = None
    plugins: list[RuntimePlugin] = field(default_factory=list)
    state_manager: StateManager = field(default_factory=StateManager)
    state: RuntimeState = field(default_factory=RuntimeState)
    _optimizer: torch.optim.Optimizer | None = field(default=None, init=False, repr=False)
    _scheduler: torch.optim.lr_scheduler.LRScheduler | None = field(default=None, init=False, repr=False)
    _step_runner: "StepRunner | None" = field(default=None, init=False, repr=False)
    _group_manager: ProcessGroupManager | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _omitted_module_paths: set[str] = field(default_factory=set, init=False, repr=False)
    _module_replacements: "dict[type[nn.Module], set[type[nn.Module]]]" = field(
        default_factory=dict, init=False
    )
    _param_roles: dict[int, ParamRole] = field(default_factory=dict, init=False, repr=False)
    _optimizer_owner: OptimizerOwner = field(init=False, repr=False)
    _step_runner_owner: StepRunnerOwner = field(init=False, repr=False)
    _model_state_owner: ModelStateOwner = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")
        if self.grad_clip_max_norm is not None and self.grad_clip_max_norm <= 0:
            raise ValueError(f"grad_clip_max_norm must be > 0, got {self.grad_clip_max_norm}")
        self.state.step_context = StepContext(
            grad_accum_steps=self.grad_accum_steps,
        )
        self._group_manager = ProcessGroupManager.from_plan(self.plan, self.mesh)
        self.plugins = self._resolve_plugin_order(self.plugins)
        self._optimizer_owner = self._resolve_optimizer_owner()
        self._step_runner_owner = self._resolve_step_runner_owner()
        self._model_state_owner = self._resolve_model_state_owner()

    def _populate_static_model_metrics(self) -> None:
        self.state.static_metrics["perf/world_size"] = self.mesh.world_size
        if isinstance(self.model, FlopsEstimatableModule):
            try:
                self.state.static_metrics["perf/flops_per_token"] = float(self.model.flops_per_token())
            except AttributeError:
                pass

    def setup(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        load_weights_only: bool = False,
    ) -> None:
        if load_weights_only and checkpoint_path is None:
            raise ValueError("load_weights_only=True requires checkpoint_path")
        if self.device is None:
            self.device = _module_device(self.model)
        else:
            self.device = torch.device(self.device)
        self.state_manager.bind(self)
        self._param_roles.clear()
        for plugin in self.plugins:
            plugin.bind(self)
        self._run_setup_phase(SetupPhase.TRANSFORM)
        self._run_setup_phase(SetupPhase.MATERIALIZE)
        self.state_manager.register_module(self.model)
        self._run_setup_phase(SetupPhase.FINALIZE)
        for plugin in self.plugins:
            plugin.annotate_param_metadata()
        self._resolve_param_checkpoint_metadata()
        self._populate_static_model_metrics()
        self._setup_optimizer_and_scheduler()
        self._step_runner = self._step_runner_owner.build_step_runner()
        if self._step_runner is None:
            if self._step_runner_owner is self:
                raise ValueError("runtime default step runner owner did not provide a step runner")
            owner_plugin = self._step_runner_owner
            raise ValueError(
                f"plugin={owner_plugin.id.value} declared owns_step_runner=True but did not provide a step runner"
            )
        if checkpoint_path is not None:
            load_sharded_checkpoint(
                self.state_manager,
                checkpoint_path,
                weights_only=load_weights_only,
            )

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        try:
            for plugin in reversed(self.plugins):
                try:
                    plugin.close()
                except BaseException as exc:
                    errors.append(exc)
        finally:
            if self._group_manager is not None:
                try:
                    self._group_manager.close()
                except BaseException as exc:
                    errors.append(exc)
            self._closed = True
        if errors:
            raise ExceptionGroup("RuntimeCore.close() failed", errors)

    def run_step(self, batch: Any) -> tuple[torch.Tensor, bool]:
        assert self.device is not None, "Runtime device should not be None!"
        assert self._step_runner is not None, "RuntimeCore step runner is not initialized; call setup() first"
        batch = _move_to_device(batch, torch.device(self.device))
        self.state.batch = batch
        num_tokens = _count_batch_tokens(batch)
        if num_tokens is not None:
            self.state.metadata["tokens"] = num_tokens
        context = self.state.step_context
        self._run_step_phase(RuntimePhase.PRE_STEP_RUNNER)
        loss = self._step_runner.run(self, self.state.batch)
        should_step = context.advance_micro_step()
        return loss, should_step

    def _run_setup_phase(self, phase: SetupPhase) -> None:
        if phase is SetupPhase.MATERIALIZE and self._model_state_owner is self:
            assert self.device is not None
            if self.dtype is None:
                self.model.to(self.device)
            else:
                self.model.to(device=self.device, dtype=self.dtype)
        for plugin in self.plugins:
            self.model = plugin.on_setup_phase(phase, self.model)

    def export_model_state(self, state_manager: StateManager) -> tuple[dict[str, torch.Tensor], list[ModelStateMeta]]:
        rank = dist.get_rank() if dist.is_initialized() else 0
        entries = [entry.meta for entry in state_manager.params.values()]
        entries.extend(entry.meta for entry in state_manager.buffers.values())
        state: dict[str, torch.Tensor] = {}
        for entry in entries:
            if entry.source_rank is None:
                raise RuntimeError(f"checkpoint source rank not resolved for model state={entry.state_key}")
            if entry.source_rank != rank:
                continue
            tensor = state_manager.get_model_tensor(entry.state_key)
            state[entry.state_key] = tensor.detach().cpu().clone()
        return state, entries

    def import_model_state(self, state_manager: StateManager, model_state: dict[str, torch.Tensor]) -> None:
        for name, tensor in model_state.items():
            resolved_name = state_manager._resolve_runtime_name(name)
            if resolved_name not in state_manager.params and resolved_name not in state_manager.buffers:
                raise KeyError(f"model state not found: {resolved_name}")
            target = state_manager.get_model_tensor(resolved_name)
            target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))

    def _run_step_phase(self, phase: RuntimePhase) -> None:
        for plugin in self.plugins:
            plugin.on_step_phase(phase)

    def get_group(self, axis: MeshAxis) -> dist.ProcessGroup | None:
        assert self._group_manager is not None
        return self._group_manager.get_group(axis)

    # -------------------------------
    # Plugin coordination state: begin
    # -------------------------------
    # Cross-plugin module-path channel: upstream transforms (for example PP)
    # mark paths they have logically removed so later plugins can skip them.
    def mark_module_path_omitted(self, path: str) -> None:
        self._omitted_module_paths.add(path)

    def is_module_path_omitted(self, path: str) -> bool:
        for omitted in self._omitted_module_paths:
            if path == omitted or path.startswith(omitted + "."):
                return True
        return False

    # Cross-plugin module-type channel: transforms record which concrete types
    # replaced a base module type so later plugins can discover them without
    # importing plugin-private layer classes directly.
    def add_module_replacement(
        self, original: "type[nn.Module]", replacement: "type[nn.Module]"
    ) -> None:
        if original not in self._module_replacements:
            self._module_replacements[original] = set()
        self._module_replacements[original].add(replacement)

    def get_module_replacements(self, original: "type[nn.Module]") -> "set[type[nn.Module]]":
        return self._module_replacements.get(original, set())

    # Cross-plugin param-semantics channel: plugins annotate whether a param is
    # shared or expert so later plugins can choose the right communication path.
    def set_param_role(self, param: nn.Parameter, role: ParamRole) -> None:
        self._param_roles[id(param)] = role

    def get_param_role(self, param: nn.Parameter) -> ParamRole:
        return self._param_roles.get(id(param), ParamRole.SHARED)

    # -------------------------------
    # Plugin coordination state: end
    # -------------------------------

    def grad_norm_replica_factor(self, param: nn.Parameter) -> int:
        fq_name = self.state_manager.get_param_name(param)
        attrs = self.state_manager.params[fq_name].attrs
        factor = 1
        for axis in attrs.replicated_axes:
            if axis == MeshAxis.DP:
                factor *= self.mesh.dp
            elif axis == MeshAxis.TP:
                factor *= self.mesh.tp
            elif axis == MeshAxis.PP:
                factor *= self.mesh.pp
            elif axis == MeshAxis.CP:
                factor *= self.mesh.cp
            else:
                group = self.get_group(axis)
                if group is not None:
                    factor *= int(dist.get_world_size(group))
        if factor < 1:
            raise ValueError(f"invalid grad_norm_replica_factor={factor} for param={fq_name}")
        return factor

    def model_state_checkpoint_rank(self, fq_name: str, *, rank_id: int) -> int:
        if fq_name in self.state_manager.buffers:
            return rank_id
        attrs = self.state_manager.params[fq_name].attrs
        return self.mesh.infer_checkpoint_rank(
            rank_id,
            plan=self.plan,
            replicated_axes=attrs.replicated_axes,
        )

    def _resolve_param_checkpoint_metadata(self) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        for fq_name in self.state_manager.param_states:
            self.state_manager.update_model_state(
                fq_name,
                source_rank=self.model_state_checkpoint_rank(fq_name, rank_id=rank),
            )
        for fq_name in self.state_manager.buffer_states:
            self.state_manager.update_model_state(
                fq_name,
                source_rank=self.model_state_checkpoint_rank(fq_name, rank_id=rank),
            )

    # -------------------------------------
    # Optimizer / scheduler methods: begin
    # -------------------------------------
    def _setup_optimizer_and_scheduler(self) -> None:
        if self._optimizer_owner is not self:
            optimizer, _ = self._optimizer_owner.get_optimizer_and_scheduler()
            if optimizer is None:
                owner_plugin = self._optimizer_owner
                raise ValueError(
                    f"plugin={owner_plugin.id.value} declared owns_optimizer=True but did not provide an optimizer"
                )
            return
        self._optimizer = self.create_optimizer(self.model.parameters())
        self._scheduler = self.create_scheduler(self._optimizer)

    def get_optimizer_and_scheduler(
        self,
    ) -> tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LRScheduler | None]:
        if self._optimizer_owner is self:
            return self._optimizer, self._scheduler
        return self._optimizer_owner.get_optimizer_and_scheduler()

    def create_optimizer(self, params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
        if self.optimizer_factory is None:
            raise ValueError("optimizer_factory is required to create an optimizer")
        params = list(params)
        if self.dtype is None or self.dtype == torch.float32:
            return self.optimizer_factory(params)
        return MasterWeightsOptimizer(params, optimizer_factory=self.optimizer_factory)

    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.scheduler_factory is None:
            return None
        return self.scheduler_factory(optimizer)

    def step_optimizer(self) -> None:
        self._run_step_phase(RuntimePhase.PRE_STEP)
        optimizer, scheduler = self.get_optimizer_and_scheduler()
        if optimizer is None:
            raise RuntimeError("step_optimizer() requires a runtime-owned or plugin-owned optimizer")
        copy_master_params = getattr(optimizer, "copy_master_params", None)
        if callable(copy_master_params):
            copy_master_params()
        scaler = self.state.scaler
        if scaler is not None:
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            next_scale = scaler.get_scale()
            self.state.metadata["loss_scale"] = float(next_scale)
            self.state.metadata["overflow"] = bool(next_scale < prev_scale)
        else:
            optimizer.step()
            self.state.metadata["loss_scale"] = None
            self.state.metadata["overflow"] = False
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        self._run_step_phase(RuntimePhase.POST_STEP)
        self.state.step_context.advance_step()

    def should_save_optimizer(self, rank_id: int) -> bool:
        return self.optimizer_checkpoint_rank(rank_id) == rank_id

    def optimizer_checkpoint_rank(self, rank_id: int) -> int:
        if self._optimizer_owner is not self:
            return self._optimizer_owner.optimizer_state_source_rank(rank_id)
        return self.optimizer_state_source_rank(rank_id)

    def optimizer_state_source_rank(self, rank_id: int) -> int:
        if any(plugin.id in {PluginId.ZERO1, PluginId.ZERO2, PluginId.ZERO3} for plugin in self.plugins):
            return rank_id

        has_shared = False
        has_expert = False
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            role = self.get_param_role(param)
            if role == ParamRole.EXPERT:
                has_expert = True
            else:
                has_shared = True
            if has_shared and has_expert:
                return rank_id
        if has_shared:
            return self.mesh.infer_checkpoint_rank(
                rank_id,
                plan=self.plan,
                replicated_axes={MeshAxis.DCP},
            )
        if has_expert:
            return self.mesh.infer_checkpoint_rank(
                rank_id,
                plan=self.plan,
                replicated_axes={MeshAxis.EREP},
            )
        return rank_id

    # -------------------------------------
    # Optimizer / scheduler methods: end
    # -------------------------------------

    def collect_metrics(self) -> dict[str, MetricValue]:
        context = self.state.step_context
        metrics: dict[str, MetricValue] = {
            "step": context.step,
        }
        metrics.update(self.state.static_metrics)
        if self.state.loss is not None:
            metrics["loss"] = float(self.state.loss.detach().float().item())
        tokens = self.state.metadata.get("tokens")
        if tokens is not None:
            metrics["train/tokens"] = _global_token_contribution(int(tokens), self.mesh)

        optimizer, scheduler = self.get_optimizer_and_scheduler()
        if optimizer is not None and optimizer.param_groups:
            metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            metrics["scheduler_last_epoch"] = int(scheduler.last_epoch)

        for plugin in self.plugins:
            for key, value in plugin.collect_metrics().items():
                metric_key = key if "/" in key else f"{plugin.id.value}/{key}"
                if metric_key in metrics:
                    raise ValueError(f"duplicate metric key={metric_key}")
                metrics[metric_key] = value
        return metrics

    def build_step_runner(self) -> "StepRunner | None":
        return DefaultStepRunner()

    def _resolve_step_runner_owner(self) -> StepRunnerOwner:
        owner_plugin: RuntimePlugin | None = None
        stray_runners: list[str] = []
        for plugin in self.plugins:
            runner = plugin.build_step_runner()
            if plugin.owns_step_runner:
                if owner_plugin is not None:
                    names = [owner_plugin.id.value, plugin.id.value]
                    raise ValueError(f"RuntimeCore allows only one step-runner-owning plugin, got {names}")
                owner_plugin = plugin
                continue
            if runner is not None:
                stray_runners.append(plugin.id.value)

        if stray_runners:
            warnings.warn(
                "plugins returned step runners without declaring owns_step_runner=True; "
                f"their step runners will be ignored: {stray_runners}",
                stacklevel=2,
            )
        return self if owner_plugin is None else owner_plugin

    def _resolve_optimizer_owner(self) -> OptimizerOwner:
        owner_plugin: RuntimePlugin | None = None
        for plugin in self.plugins:
            if not plugin.owns_optimizer:
                continue
            if owner_plugin is not None:
                raise ValueError(
                    "RuntimeCore allows only one optimizer-owning plugin, "
                    f"got {[owner_plugin.id.value, plugin.id.value]}"
                )
            owner_plugin = plugin
        return self if owner_plugin is None else owner_plugin

    def _resolve_model_state_owner(self) -> ModelStateOwner:
        owner_plugin: RuntimePlugin | None = None
        for plugin in self.plugins:
            if not plugin.owns_model_state:
                continue
            if owner_plugin is not None:
                raise ValueError(
                    "RuntimeCore allows only one model-state-owning plugin, "
                    f"got {[owner_plugin.id.value, plugin.id.value]}"
                )
            owner_plugin = plugin
        return self if owner_plugin is None else owner_plugin

    def _resolve_plugin_order(self, plugins: list[RuntimePlugin]) -> list[RuntimePlugin]:
        if not plugins:
            return []
        plugin_by_id = {}
        for plugin in plugins:
            if plugin.id in plugin_by_id:
                raise ValueError(f"duplicate plugin id={plugin.id.value}")
            plugin_by_id[plugin.id] = plugin

        sorter = TopologicalSorter()
        for plugin in plugins:
            missing = set(plugin.requires) - set(plugin_by_id)
            if missing:
                missing_values = sorted(plugin_id.value for plugin_id in missing)
                raise ValueError(f"plugin={plugin.name} requires missing plugins={missing_values}")
            deps = set(plugin.requires) | (set(plugin.runs_after) & set(plugin_by_id))
            sorter.add(plugin.id, *deps)
        for plugin in plugins:
            for before_name in plugin.runs_before:
                if before_name in plugin_by_id:
                    sorter.add(before_name, plugin.id)
        order = [plugin_id for plugin_id in sorter.static_order() if plugin_id in plugin_by_id]
        return [plugin_by_id[plugin_id] for plugin_id in order]


def _count_batch_tokens(batch: Any) -> int | None:
    if isinstance(batch, dict):
        input_ids = batch.get(INPUT_IDS_KEY)
        if torch.is_tensor(input_ids):
            return int(input_ids.numel())
    if isinstance(batch, (tuple, list)) and batch and torch.is_tensor(batch[0]):
        return int(batch[0].numel())
    if torch.is_tensor(batch):
        return int(batch.numel())
    return None


def _module_device(module: nn.Module) -> torch.device:
    first_param = next(module.parameters(), None)
    if first_param is not None:
        return first_param.device
    first_buffer = next(module.buffers(), None)
    if first_buffer is not None:
        return first_buffer.device
    return torch.device("cpu")


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _global_token_contribution(local_tokens: int, mesh: MeshConfig) -> float:
    replicated_ranks = mesh.tp * mesh.pp * mesh.cp
    return float(local_tokens) / float(replicated_ranks)
