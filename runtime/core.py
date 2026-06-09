from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from graphlib import TopologicalSorter
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist

from parallel.plan import ParallelPlan
from runtime.mesh import MeshAxis, MeshConfig, ProcessGroupManager
from runtime.plugin import FlopsEstimatableModule, MetricValue, RuntimePlugin
from state.state import StateManager

OptimizerFactory = Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]
SchedulerFactory = Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
StepRunnerFn = Callable[[Any], torch.Tensor]


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


class ParamRole(str, Enum):
    SHARED = "shared"
    EXPERT = "expert"


class PpStatus(str, Enum):
    IDLE = "idle"
    FORWARD = "forward"
    BACKWARD_START = "backward_start"
    BACKWARD_MIDDLE = "backward_middle"
    BACKWARD_END = "backward_end"


@dataclass
class StepContext:
    step: int = 0
    microbatch_idx: int = 0
    grad_accum_steps: int = 1
    pp_cur_microbatch_idx: int = 0
    pp_status: PpStatus = PpStatus.IDLE

    def __post_init__(self) -> None:
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")

    @property
    def accum_start(self) -> bool:
        return self.microbatch_idx == 0 and self.pp_status in {PpStatus.IDLE, PpStatus.BACKWARD_START}

    @property
    def is_step_boundary(self) -> bool:
        return (
            ((self.microbatch_idx + 1) % self.grad_accum_steps) == 0
            and self.pp_status in {PpStatus.IDLE, PpStatus.BACKWARD_END}
        )

    @property
    def loss_divisor(self) -> float:
        return float(self.grad_accum_steps)

    def set_pp_state(self, *, microbatch_idx: int, status: PpStatus) -> None:
        if microbatch_idx < 0:
            raise ValueError(f"pp_cur_microbatch_idx must be >= 0, got {microbatch_idx}")
        self.pp_cur_microbatch_idx = microbatch_idx
        self.pp_status = status

    def advance_micro_step(self) -> bool:
        should_step = self.is_step_boundary
        self.microbatch_idx = (self.microbatch_idx + 1) % self.grad_accum_steps
        self.pp_cur_microbatch_idx = 0
        self.pp_status = PpStatus.IDLE
        return should_step

    def advance_step(self) -> None:
        self.step += 1


@dataclass
class RuntimeState:
    """Transient execution context for the current step.

    This is intentionally not a checkpoint manifest. Plugins use it as a
    shared scratchpad for phase-local data such as the current batch, loss,
    microbatch index, profiler annotations, or temporary scheduling metadata.
    Durable training state should be declared through checkpoint/state APIs.
    """

    step_context: StepContext = field(default_factory=StepContext)
    loss: torch.Tensor | None = None
    batch: Any = None
    outputs: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    static_metrics: dict[str, MetricValue] = field(default_factory=dict)
    scaler: torch.amp.GradScaler | None = None
    omitted_module_paths: set[str] = field(default_factory=set)
    param_roles: dict[int, ParamRole] = field(default_factory=dict)

    @property
    def step(self) -> int:
        return self.step_context.step

    @step.setter
    def step(self, value: int) -> None:
        self.step_context.step = value


@dataclass
class RuntimeCore:
    model: nn.Module
    mesh: MeshConfig = field(default_factory=MeshConfig)
    plan: ParallelPlan = field(default_factory=ParallelPlan)
    device: torch.device | str | None = None
    grad_accum_steps: int = 1
    optimizer_factory: OptimizerFactory | None = None
    scheduler_factory: SchedulerFactory | None = None
    plugins: list[RuntimePlugin] = field(default_factory=list)
    group_manager: ProcessGroupManager | None = None
    state_manager: StateManager = field(default_factory=StateManager)
    state: RuntimeState = field(default_factory=RuntimeState)
    optimizer: torch.optim.Optimizer | None = field(default=None, init=False)
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = field(default=None, init=False)
    _post_dp_reduction_callbacks: list[
        tuple[Callable[[torch.Tensor], "dist.Work | None"], "ParamRole | None"]
    ] = field(default_factory=list, init=False)

    def register_post_dp_reduction_callback(
        self,
        cb: Callable[[torch.Tensor], "dist.Work | None"],
        *,
        role_filter: "ParamRole | None" = None,
    ) -> None:
        """Register a callback invoked after each ZeRO bucket's DP reduction completes.

        Plugins (e.g. CP) use this instead of embedding cross-plugin sync logic
        directly into ZeRO. The callback receives the local gradient shard and
        may return an async Work handle; ZeRO will wait on it before PRE_STEP.

        The callback must be tensor-agnostic: it receives a flat 1D shard whose
        shape and parameter identity vary per bucket. Selective per-parameter
        logic is not supported.

        role_filter: if set, ZeRO only invokes this callback for buckets whose
        ParamRole matches. Use ParamRole.EXPERT for expert-only grad syncs.
        """
        self._post_dp_reduction_callbacks.append((cb, role_filter))

    def __post_init__(self) -> None:
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}")
        self.state.step_context = StepContext(
            grad_accum_steps=self.grad_accum_steps,
        )
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
        self._populate_static_model_metrics()
        self._maybe_build_runtime_optimizer()
        self._validate_optimizer_owner()
    def close(self) -> None:
        for plugin in reversed(self.plugins):
            plugin.close()

    def run_step(self, batch: Any) -> tuple[torch.Tensor, bool]:
        if self.device is not None:
            batch = _move_to_device(batch, torch.device(self.device))
        self.state.batch = batch
        tokens = _count_batch_tokens(batch)
        if tokens is not None:
            self.state.metadata["tokens"] = tokens
        context = self.state.step_context
        self._run_phase(RuntimePhase.PRE_MICROBATCH)
        loss = self.get_step_runner()(self.state.batch)
        should_step = context.advance_micro_step()
        return loss, should_step

    def _run_step_impl(self, batch: Any) -> torch.Tensor:
        self._forward_step_impl(batch)
        if not torch.is_tensor(self.state.loss):
            raise TypeError("RuntimeCore expects model(batch) to return a Tensor loss during training.")
        self._backward_step_impl()
        assert self.state.loss is not None
        return self.state.loss

    def _forward_step_impl(self, batch: Any) -> None:
        self._run_phase(RuntimePhase.PRE_FORWARD)
        outputs = self.model(batch)
        self.state.outputs = outputs
        self.state.loss = outputs if torch.is_tensor(outputs) else None
        self._run_phase(RuntimePhase.POST_FORWARD)

    def _backward_step_impl(
        self,
        *,
        grad_output: torch.Tensor | None = None,
    ) -> None:
        self._run_phase(RuntimePhase.PRE_BACKWARD)
        if grad_output is None:
            if self.state.loss is None:
                raise TypeError("RuntimeCore expected self.state.loss to be a Tensor before backward()")
            divisor = self.state.step_context.loss_divisor
            if divisor != 1:
                self.state.loss = self.state.loss / divisor
            self.state.loss.backward()
        else:
            if not torch.is_tensor(self.state.outputs):
                raise TypeError("RuntimeCore expected self.state.outputs Tensor for activation backward()")
            self.state.outputs.backward(grad_output)
        self._run_phase(RuntimePhase.POST_BACKWARD)

    def get_group(self, axis: MeshAxis) -> dist.ProcessGroup | None:
        assert self.group_manager is not None
        return self.group_manager.get_group(axis)

    def mark_module_path_omitted(self, path: str) -> None:
        self.state.omitted_module_paths.add(path)

    def is_module_path_omitted(self, path: str) -> bool:
        for omitted in self.state.omitted_module_paths:
            if path == omitted or path.startswith(omitted + "."):
                return True
        return False

    def set_param_role(self, param: nn.Parameter, role: ParamRole) -> None:
        self.state.param_roles[id(param)] = role

    def get_param_role(self, param: nn.Parameter) -> ParamRole:
        return self.state.param_roles.get(id(param), ParamRole.SHARED)

    def get_optimizer_and_scheduler(
        self,
    ) -> tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LRScheduler | None]:
        if self.optimizer is not None:
            return self.optimizer, self.scheduler
        for plugin in self.plugins:
            if plugin.owns_optimizer:
                return getattr(plugin, "optimizer", None), getattr(plugin, "scheduler", None)
        return None, None

    def create_optimizer(self, params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
        if self.optimizer_factory is None:
            raise ValueError("optimizer_factory is required to create an optimizer")
        return self.optimizer_factory(params)

    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        if self.scheduler_factory is None:
            return None
        return self.scheduler_factory(optimizer)

    def get_step_runner(self) -> StepRunnerFn:
        runners = []
        for plugin in self.plugins:
            runner = plugin.build_step_runner()
            if runner is not None:
                runners.append((plugin.id.value, runner))
        if len(runners) > 1:
            names = [name for name, _ in runners]
            raise ValueError(f"RuntimeCore allows only one step runner plugin, got {names}")
        if runners:
            return runners[0][1]
        return self._run_step_impl

    def step_optimizer(self) -> None:
        self._run_phase(RuntimePhase.PRE_STEP)
        optimizer, scheduler = self.get_optimizer_and_scheduler()
        if optimizer is None:
            raise RuntimeError("step_optimizer() requires a runtime-owned or plugin-owned optimizer")
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
        self._run_phase(RuntimePhase.POST_STEP)
        self.state.step_context.advance_step()

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

    def should_save_optimizer(self, rank_id: int) -> bool:
        return self.optimizer_state_source_rank(rank_id) == rank_id

    def optimizer_state_source_rank(self, rank_id: int) -> int:
        if self._plugin_owns_optimizer():
            for plugin in self.plugins:
                if plugin.owns_optimizer:
                    return plugin.optimizer_state_source_rank(rank_id)
            raise RuntimeError("optimizer ownership invariant violated: no optimizer-owning plugin found")

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
        runtime_owners = ["runtime"] if self.optimizer is not None else []
        plugin_owners = [plugin.id.value for plugin in self.plugins if plugin.owns_optimizer]
        if len(plugin_owners) > 1:
            raise ValueError(f"RuntimeCore allows only one optimizer-owning plugin, got {plugin_owners}")
        if runtime_owners and plugin_owners:
            raise ValueError(
                "Runtime optimizer ownership is mutually exclusive with optimizer-owning plugins, "
                f"got {runtime_owners + plugin_owners}"
            )

    def _maybe_build_runtime_optimizer(self) -> None:
        if self._plugin_owns_optimizer():
            return
        self.optimizer = self.create_optimizer(self.model.parameters())
        self.scheduler = self.create_scheduler(self.optimizer)

    def _populate_static_model_metrics(self) -> None:
        self.state.static_metrics["perf/world_size"] = self.mesh.world_size
        if isinstance(self.model, FlopsEstimatableModule):
            try:
                self.state.static_metrics["perf/flops_per_token"] = float(self.model.flops_per_token())
            except AttributeError:
                pass

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
