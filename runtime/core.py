from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from graphlib import TopologicalSorter
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist

from parallel.plan import ParallelPlan
from runtime.mesh import MeshAxis, MeshConfig, ProcessGroupManager
from runtime.plugin import MetricValue, RuntimePlugin
from state.state import StateManager

OptimizerFactory = Callable[[nn.Module], torch.optim.Optimizer]
SchedulerFactory = Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]


class RuntimePhase(str, Enum):
    SETUP = "setup"
    TRANSFORM_MODEL = "transform_model"
    PRE_MICROBATCH = "pre_microbatch"
    PRE_FORWARD = "pre_forward"
    POST_FORWARD = "post_forward"
    PRE_BACKWARD = "pre_backward"
    POST_BACKWARD = "post_backward"
    PRE_STEP = "pre_step"
    POST_STEP = "post_step"
    PRE_SAVE = "pre_save"
    POST_LOAD = "post_load"


@dataclass
class RuntimeState:
    """Transient execution context for the current step.

    This is intentionally not a checkpoint manifest. Plugins use it as a
    shared scratchpad for phase-local data such as the current batch, loss,
    microbatch index, profiler annotations, or temporary scheduling metadata.
    Durable training state should be declared through checkpoint/state APIs.
    """

    step: int = 0
    microbatch_idx: int = 0
    loss: torch.Tensor | None = None
    batch: Any = None
    outputs: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    scaler: torch.amp.GradScaler | None = None


@dataclass
class RuntimeCore:
    model: nn.Module
    mesh: MeshConfig = field(default_factory=MeshConfig)
    plan: ParallelPlan = field(default_factory=ParallelPlan)
    device: torch.device | str | None = None
    optimizer: torch.optim.Optimizer | None = None
    optimizer_factory: OptimizerFactory | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    scheduler_factory: SchedulerFactory | None = None
    plugins: list[RuntimePlugin] = field(default_factory=list)
    group_manager: ProcessGroupManager | None = None
    state_manager: StateManager = field(default_factory=StateManager)
    state: RuntimeState = field(default_factory=RuntimeState)
    grad_accum_steps: int = 1

    def __post_init__(self) -> None:
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")
        self._validate_mesh_and_plan()
        if self.group_manager is None:
            self.group_manager = ProcessGroupManager.from_mesh(self.mesh)
        self.plugins = self._resolve_plugin_order(self.plugins)

    def setup(self) -> None:
        if self.device is not None:
            self.device = torch.device(self.device)
            self.model.to(self.device)
        self.state_manager.bind(self)
        for plugin in self.plugins:
            plugin.bind(self)
        self._run_phase(RuntimePhase.SETUP)
        for plugin in self.plugins:
            self.model = plugin.transform_model(self.model)
        self.state_manager.register_module(self.model)
        self._run_phase(RuntimePhase.TRANSFORM_MODEL)
        self._build_runtime_optimizer()
        self._validate_optimizer_owner()

    def run_train_step(self, batch: Any) -> torch.Tensor:
        if self.device is not None:
            batch = _move_to_device(batch, torch.device(self.device))
        self.state.batch = batch
        tokens = _count_batch_tokens(batch)
        if tokens is not None:
            self.state.metadata["tokens"] = tokens
        accum_idx = self.state.microbatch_idx
        should_step = ((accum_idx + 1) % self.grad_accum_steps) == 0
        accum_start = accum_idx == 0
        self.state.metadata["should_sync_grad"] = should_step
        self.state.metadata["should_step_optimizer"] = should_step
        self.state.metadata["accum_start"] = accum_start
        self._run_phase(RuntimePhase.PRE_MICROBATCH)
        self._run_phase(RuntimePhase.PRE_FORWARD)
        outputs = self.model(batch)
        self.state.outputs = outputs
        self.state.loss = outputs if torch.is_tensor(outputs) else None
        self._run_phase(RuntimePhase.POST_FORWARD)
        self._run_phase(RuntimePhase.PRE_BACKWARD)
        if self.state.loss is None:
            raise TypeError("RuntimeCore expects model(batch) to return a Tensor loss during training.")
        if self.grad_accum_steps > 1:
            self.state.loss = self.state.loss / self.grad_accum_steps
        self.state.loss.backward()
        self._run_phase(RuntimePhase.POST_BACKWARD)
        if should_step:
            self._run_phase(RuntimePhase.PRE_STEP)
            self.optimizer_step()
            self._run_phase(RuntimePhase.POST_STEP)
            self.state.step += 1
        self.state.microbatch_idx = (accum_idx + 1) % self.grad_accum_steps
        return self.state.loss

    def get_group(self, axis: MeshAxis) -> dist.ProcessGroup | None:
        assert self.group_manager is not None
        return self.group_manager.get_group(axis)

    def get_optimizer_and_scheduler(
        self,
    ) -> tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LRScheduler | None]:
        if self.optimizer is not None:
            return self.optimizer, self.scheduler
        for plugin in self.plugins:
            if plugin.owns_optimizer:
                return getattr(plugin, "optimizer", None), getattr(plugin, "scheduler", None)
        return None, None

    def optimizer_step(self) -> None:
        optimizer, scheduler = self.get_optimizer_and_scheduler()
        if optimizer is None:
            return
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

    def collect_metrics(self) -> dict[str, MetricValue]:
        metrics: dict[str, MetricValue] = {
            "step": self.state.step,
            "microbatch_idx": self.state.microbatch_idx,
            "grad_accum_steps": self.grad_accum_steps,
            "should_step_optimizer": bool(self.state.metadata.get("should_step_optimizer", True)),
        }
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

    def should_save_optimizer(self, rank_id: int) -> bool:
        return self.optimizer_state_source_rank(rank_id) == rank_id

    def optimizer_state_source_rank(self, rank_id: int) -> int:
        if self.optimizer is None:
            for plugin in self.plugins:
                if plugin.owns_optimizer:
                    return plugin.optimizer_state_source_rank(rank_id)
            return rank_id

        replicated_axes, sharded_axes = self._runtime_optimizer_mesh_axes()
        dp_idx, pp_idx, cp_idx, tp_idx = self.mesh.rank_coordinates(rank_id)
        coords = {
            MeshAxis.DP: dp_idx,
            MeshAxis.PP: pp_idx,
            MeshAxis.CP: cp_idx,
            MeshAxis.TP: tp_idx,
        }
        for axis in replicated_axes - sharded_axes:
            coords[axis] = 0
        return self.mesh.rank_id(
            dp=coords[MeshAxis.DP],
            pp=coords[MeshAxis.PP],
            cp=coords[MeshAxis.CP],
            tp=coords[MeshAxis.TP],
        )

    def _run_phase(self, phase: RuntimePhase) -> None:
        for plugin in self.plugins:
            plugin.on_phase(phase)

    def _plugin_owns_optimizer(self) -> bool:
        return any(plugin.owns_optimizer for plugin in self.plugins)

    def _runtime_optimizer_mesh_axes(self) -> tuple[set[MeshAxis], set[MeshAxis]]:
        replicated_axes: set[MeshAxis] = set()
        sharded_axes: set[MeshAxis] = set()
        for plugin in self.plugins:
            replicated_axes.update(plugin.runtime_optimizer_replicated_axes())
            sharded_axes.update(plugin.runtime_optimizer_sharded_axes())
        return replicated_axes, sharded_axes

    def _validate_optimizer_owner(self) -> None:
        runtime_owners = [name for name, value in {
            "runtime_optimizer": self.optimizer,
            "runtime_optimizer_factory": self.optimizer_factory,
        }.items() if value is not None]
        if len(runtime_owners) > 1:
            raise ValueError(f"RuntimeCore allows only one runtime optimizer source, got {runtime_owners}")
        if self.scheduler_factory is not None and not runtime_owners:
            raise ValueError("scheduler_factory requires a runtime optimizer or optimizer_factory")

        owners = ["runtime"] if runtime_owners else []
        plugin_owners = [plugin.id.value for plugin in self.plugins if plugin.owns_optimizer]
        if len(plugin_owners) > 1:
            raise ValueError(f"RuntimeCore allows only one optimizer-owning plugin, got {plugin_owners}")
        if runtime_owners and plugin_owners:
            raise ValueError(
                "Runtime optimizer ownership is mutually exclusive with optimizer-owning plugins, "
                f"got {runtime_owners + plugin_owners}"
            )
        owners.extend(plugin_owners)
        if len(owners) != 1:
            raise ValueError(f"RuntimeCore training requires exactly one optimizer owner, got {owners}")

    def _build_runtime_optimizer(self) -> None:
        if self.optimizer_factory is None:
            if self.scheduler_factory is not None and self.optimizer is not None:
                if self.scheduler is not None:
                    raise ValueError("scheduler_factory cannot be used when scheduler is already set")
                self.scheduler = self.scheduler_factory(self.optimizer)
            return
        if self.optimizer is not None:
            raise ValueError("optimizer_factory cannot be used when optimizer is already set")
        if self.scheduler is not None:
            raise ValueError("optimizer_factory cannot be used with a prebuilt scheduler")
        self.optimizer = self.optimizer_factory(self.model)
        if self.scheduler_factory is not None:
            self.scheduler = self.scheduler_factory(self.optimizer)
        self.optimizer_factory = None
        self.scheduler_factory = None

    def _validate_mesh_and_plan(self) -> None:
        if self.plan.zero_stage > 0 and self.mesh.dp <= 1:
            raise ValueError(f"zero_stage={self.plan.zero_stage} requires mesh.dp > 1")
        if self.mesh.pp <= 1 and self.plan.pp_schedule.virtual_stages > 1:
            raise ValueError("virtual pipeline stages require mesh.pp > 1")

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
        input_ids = batch.get("input_ids")
        if torch.is_tensor(input_ids):
            return int(input_ids.numel())
    if isinstance(batch, (tuple, list)) and batch and torch.is_tensor(batch[0]):
        return int(batch[0].numel())
    if torch.is_tensor(batch):
        return int(batch.numel())
    return None


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
