from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from graphlib import TopologicalSorter
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist

from train_system.parallel.plan import ParallelPlan
from train_system.runtime.mesh import MeshAxis, MeshConfig, ProcessGroupManager
from train_system.runtime.plugin import RuntimePlugin
from train_system.state.registry import StateRegistry


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


@dataclass
class RuntimeCore:
    model: nn.Module
    mesh: MeshConfig = field(default_factory=MeshConfig)
    plan: ParallelPlan = field(default_factory=ParallelPlan)
    optimizer: torch.optim.Optimizer | None = None
    plugins: list[RuntimePlugin] = field(default_factory=list)
    group_manager: ProcessGroupManager | None = None
    state_registry: StateRegistry = field(default_factory=StateRegistry)
    state: RuntimeState = field(default_factory=RuntimeState)

    def __post_init__(self) -> None:
        self._validate_mesh_and_plan()
        if self.group_manager is None:
            self.group_manager = ProcessGroupManager.from_mesh(self.mesh)
        self.plugins = self._resolve_plugin_order(self.plugins)
        self._validate_optimizer_owner()

    def setup(self) -> None:
        for plugin in self.plugins:
            plugin.bind(self)
        self._run_phase(RuntimePhase.SETUP)
        for plugin in self.plugins:
            self.model = plugin.transform_model(self.model)
        self.state_registry.register_module(self.model)
        self._run_phase(RuntimePhase.TRANSFORM_MODEL)

    def run_train_step(self, batch: Any) -> torch.Tensor:
        self.state.batch = batch
        self._run_phase(RuntimePhase.PRE_MICROBATCH)
        self._run_phase(RuntimePhase.PRE_FORWARD)
        outputs = self.model(batch)
        self.state.outputs = outputs
        self.state.loss = outputs if torch.is_tensor(outputs) else None
        self._run_phase(RuntimePhase.POST_FORWARD)
        self._run_phase(RuntimePhase.PRE_BACKWARD)
        if self.state.loss is None:
            raise TypeError("RuntimeCore expects model(batch) to return a Tensor loss during training.")
        self.state.loss.backward()
        self._run_phase(RuntimePhase.POST_BACKWARD)
        self._run_phase(RuntimePhase.PRE_STEP)
        if self.optimizer is not None and not self._plugin_owns_optimizer():
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        self._run_phase(RuntimePhase.POST_STEP)
        self.state.step += 1
        return self.state.loss

    def get_group(self, axis: MeshAxis) -> dist.ProcessGroup | None:
        assert self.group_manager is not None
        return self.group_manager.get_group(axis)

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
        owners = ["runtime"] if self.optimizer is not None else []
        plugin_owners = [plugin.id.value for plugin in self.plugins if plugin.owns_optimizer]
        if len(plugin_owners) > 1:
            raise ValueError(f"RuntimeCore allows only one optimizer-owning plugin, got {plugin_owners}")
        if self.optimizer is not None and plugin_owners:
            raise ValueError(
                "RuntimeCore.optimizer is mutually exclusive with optimizer-owning plugins, "
                f"got {['runtime', *plugin_owners]}"
            )
        owners.extend(plugin_owners)
        if len(owners) != 1:
            raise ValueError(f"RuntimeCore training requires exactly one optimizer owner, got {owners}")

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
