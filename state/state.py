from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from runtime.mesh import MeshAxis

if TYPE_CHECKING:
    from data import StatefulDataLoaderProtocol
    from runtime.core import RuntimeCore


@dataclass
class RuntimeParamStatus:
    """Runtime-only mutable state for one parameter reference."""

    is_materialized: bool = True
    is_gathered: bool = False


@dataclass
class RuntimeParamLayout:
    """Runtime-only distributed layout metadata for one parameter reference."""

    logical_shape: tuple[int, ...]
    sharded_axes: set[MeshAxis] = field(default_factory=set)
    replicated_axes: set[MeshAxis] = field(default_factory=set)


# TODO: RuntimeParamStatus / RuntimeParamLayout / ParamState are split
# across runtime-mutability, runtime layout, and checkpoint metadata. This is
# workable for now but easy to lose track of; later we should consolidate them
# behind a clearer single param-metadata model.


@dataclass
class ParamState:
    state_key: str
    logical_names: list[str]
    logical_shapes: list[tuple[int, ...]]
    physical_shape: tuple[int, ...]
    dtype: str
    annotations: dict[str, object] = field(default_factory=dict)

    def set_plugin_annotation(self, plugin_name: str, value: object) -> None:
        if plugin_name in self.annotations:
            raise ValueError(f"duplicate checkpoint annotation: {plugin_name}")
        self.annotations[plugin_name] = value


@dataclass(frozen=True)
class RngState:
    cpu: torch.Tensor
    cuda: list[torch.Tensor] | None = None


@dataclass(frozen=True)
class TrainerState:
    rng: RngState
    step_context: dict[str, Any] | None = None
    consumed_tokens: int | None = None
    dataloader: dict[str, Any] | None = None
    plugin_states: dict[str, dict[str, Any]] | None = None


@dataclass(frozen=True)
class OptimizerState:
    state: dict[str, Any]


@dataclass
class StateManager:
    """Runtime-owned index of logical training state."""

    param_states: dict[str, ParamState] = field(default_factory=dict)
    _runtime_params: dict[str, nn.Parameter] = field(default_factory=dict, repr=False)
    _runtime_status: dict[str, RuntimeParamStatus] = field(default_factory=dict, repr=False)
    _runtime_layouts: dict[str, RuntimeParamLayout] = field(default_factory=dict, repr=False)
    _param_names_by_id: dict[int, str] = field(default_factory=dict, repr=False)
    _runtime: "RuntimeCore | None" = field(default=None, init=False, repr=False)
    _dataloader: "StatefulDataLoaderProtocol | None" = field(default=None, init=False, repr=False)

    _OPTIMIZER_STATE_PREFIX = "__optimizer_state__."
    _RUNTIME_OPTIMIZER_STATE_KEY = f"{_OPTIMIZER_STATE_PREFIX}runtime"
    _SCHEDULER_STATE_PREFIX = "__scheduler_state__."
    _RUNTIME_SCHEDULER_STATE_KEY = f"{_SCHEDULER_STATE_PREFIX}runtime"

    def _resolve_runtime_name(self, fq_name: str) -> str:
        if fq_name in self._runtime_params or fq_name in self.param_states or fq_name in self._runtime_layouts:
            return fq_name
        with_prefix = f"module.{fq_name}"
        if with_prefix in self._runtime_params or with_prefix in self.param_states or with_prefix in self._runtime_layouts:
            return with_prefix
        if fq_name.startswith("module."):
            without_prefix = fq_name[len("module.") :]
            if without_prefix in self._runtime_params or without_prefix in self.param_states or without_prefix in self._runtime_layouts:
                return without_prefix
        return fq_name

    def register_module(self, model: nn.Module) -> None:
        self.param_states.clear()
        self._runtime_params.clear()
        self._runtime_status.clear()
        self._runtime_layouts.clear()
        self._param_names_by_id.clear()
        for fq_name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self._runtime_params[fq_name] = param
            self._param_names_by_id[id(param)] = fq_name
            self._runtime_status[fq_name] = RuntimeParamStatus(
                is_materialized=True,
                is_gathered=False,
            )
            self.param_states[fq_name] = ParamState(
                state_key=fq_name,
                logical_names=[fq_name],
                logical_shapes=[tuple(param.shape)],
                physical_shape=tuple(param.shape),
                dtype=str(param.dtype),
            )
            self._runtime_layouts[fq_name] = RuntimeParamLayout(
                logical_shape=tuple(param.shape),
            )

    def bind(self, runtime: "RuntimeCore") -> None:
        self._runtime = runtime

    @property
    def runtime(self) -> "RuntimeCore":
        if self._runtime is None:
            raise RuntimeError("StateManager is not bound to RuntimeCore")
        return self._runtime

    def bind_dataloader(self, dataloader: "StatefulDataLoaderProtocol | None") -> None:
        self._dataloader = dataloader

    def update_param_state(
        self,
        fq_name: str,
        *,
        logical_names: list[str] | None = None,
        logical_shapes: list[tuple[int, ...]] | None = None,
        physical_shape: tuple[int, ...] | None = None,
        dtype: str | None = None,
    ) -> None:
        fq_name = self._resolve_runtime_name(fq_name)
        state = self.param_states.get(fq_name)
        if state is None:
            raise KeyError(f"param state not found: {fq_name}")
        if logical_names is not None:
            state.logical_names = logical_names
        if logical_shapes is not None:
            state.logical_shapes = logical_shapes
        if physical_shape is not None:
            state.physical_shape = physical_shape
        if dtype is not None:
            state.dtype = dtype
        layout = self._runtime_layouts.get(fq_name)
        if layout is not None and logical_shapes is not None and len(logical_shapes) == 1:
            layout.logical_shape = logical_shapes[0]

    def get_param_state(self, fq_name: str) -> ParamState:
        fq_name = self._resolve_runtime_name(fq_name)
        if fq_name not in self.param_states:
            raise KeyError(f"param state not found: {fq_name}")
        return self.param_states[fq_name]

    def get_param_tensor(self, fq_name: str) -> nn.Parameter:
        fq_name = self._resolve_runtime_name(fq_name)
        if fq_name not in self._runtime_params:
            raise KeyError(f"runtime parameter not found: {fq_name}")
        return self._runtime_params[fq_name]

    def get_param_name(self, param: nn.Parameter) -> str:
        fq_name = self._param_names_by_id.get(id(param))
        if fq_name is None:
            raise KeyError("runtime parameter name not found")
        return fq_name

    def get_param_status(self, fq_name: str) -> RuntimeParamStatus:
        fq_name = self._resolve_runtime_name(fq_name)
        if fq_name not in self._runtime_status:
            raise KeyError(f"runtime status not found: {fq_name}")
        return self._runtime_status[fq_name]

    def get_param_layout(self, fq_name: str) -> RuntimeParamLayout:
        fq_name = self._resolve_runtime_name(fq_name)
        if fq_name not in self._runtime_layouts:
            raise KeyError(f"runtime param layout not found: {fq_name}")
        return self._runtime_layouts[fq_name]

    def add_param_sharded_axis(self, fq_name: str, axis: MeshAxis) -> None:
        self.get_param_layout(fq_name).sharded_axes.add(axis)

    def add_param_replicated_axis(self, fq_name: str, axis: MeshAxis) -> None:
        self.get_param_layout(fq_name).replicated_axes.add(axis)

    def iter_param_states(self):
        return self.param_states.items()

    def export_param_states(self) -> list[ParamState]:
        entries: list[ParamState] = []
        for _, state in self.iter_param_states():
            entries.append(
                ParamState(
                    state_key=state.state_key,
                    logical_names=list(state.logical_names),
                    logical_shapes=list(state.logical_shapes),
                    physical_shape=tuple(state.physical_shape),
                    dtype=state.dtype,
                    annotations=dict(state.annotations),
                )
            )
        return entries

    def export_optimizer_state(self) -> OptimizerState | None:
        if self._runtime is None:
            raise RuntimeError("StateManager is not bound to RuntimeCore")
        runtime = self._runtime
        if runtime.optimizer is not None:
            state: dict[str, Any] = {self._RUNTIME_OPTIMIZER_STATE_KEY: runtime.optimizer.state_dict()}
            if runtime.scheduler is not None:
                state[self._RUNTIME_SCHEDULER_STATE_KEY] = runtime.scheduler.state_dict()
            return OptimizerState(state=state)

        for plugin in runtime.plugins:
            if not plugin.owns_optimizer:
                continue
            optimizer = getattr(plugin, "optimizer", None)
            if optimizer is None:
                continue
            state: dict[str, Any] = {f"{self._OPTIMIZER_STATE_PREFIX}{plugin.id.value}": optimizer.state_dict()}
            scheduler = getattr(plugin, "scheduler", None)
            if scheduler is not None:
                state[f"{self._SCHEDULER_STATE_PREFIX}{plugin.id.value}"] = scheduler.state_dict()
            return OptimizerState(state=state)
        return None

    def export_model_state(self) -> tuple[dict[str, torch.Tensor], list[ParamState]]:
        if self._runtime is None:
            raise RuntimeError("StateManager is not bound to RuntimeCore")
        runtime = self._runtime
        for plugin in runtime.plugins:
            override_state = plugin.override_param_state_dict()
            if override_state is not None:
                state, entries = override_state
                break
        else:
            state = {}
            entries = self.export_param_states()
            for name, _ in self.iter_param_states():
                param = self.get_param_tensor(name)
                state[name] = param.detach().cpu().clone()

        for entry in entries:
            for plugin in runtime.plugins:
                plugin.annotate_checkpoint_state(entry)
        return state, entries

    def import_model_state(self, model_state: dict[str, torch.Tensor]) -> None:
        if self._runtime is None:
            raise RuntimeError("StateManager is not bound to RuntimeCore")
        runtime = self._runtime
        for plugin in runtime.plugins:
            if plugin.load_param_state_dict(model_state):
                return

        for name, tensor in model_state.items():
            self.get_param_state(name)
            param = self.get_param_tensor(name)
            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))

    def import_optimizer_state(self, state: OptimizerState) -> None:
        if self._runtime is None:
            raise RuntimeError("StateManager is not bound to RuntimeCore")
        runtime = self._runtime
        payload = state.state
        if runtime.optimizer is not None:
            optimizer_state = payload.get(self._RUNTIME_OPTIMIZER_STATE_KEY)
            if optimizer_state is not None:
                runtime.optimizer.load_state_dict(optimizer_state)
            scheduler_state = payload.get(self._RUNTIME_SCHEDULER_STATE_KEY)
            if scheduler_state is not None and runtime.scheduler is not None:
                runtime.scheduler.load_state_dict(scheduler_state)
            return

        for plugin in runtime.plugins:
            if not plugin.owns_optimizer:
                continue
            optimizer = getattr(plugin, "optimizer", None)
            if optimizer is None:
                continue
            optimizer_state = payload.get(f"{self._OPTIMIZER_STATE_PREFIX}{plugin.id.value}")
            if optimizer_state is None:
                continue
            optimizer.load_state_dict(optimizer_state)
            scheduler = getattr(plugin, "scheduler", None)
            scheduler_state = payload.get(f"{self._SCHEDULER_STATE_PREFIX}{plugin.id.value}")
            if scheduler is not None and scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            return
        raise ValueError("no optimizer state found")

    def export_trainer_state(self) -> TrainerState:
        if self._runtime is None:
            raise RuntimeError("StateManager is not bound to RuntimeCore")
        runtime = self._runtime
        rng_state = RngState(
            cpu=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )
        plugin_states: dict[str, dict[str, Any]] = {}
        for plugin in runtime.plugins:
            state = plugin.export_plugin_state()
            if state:
                plugin_states[plugin.id.value] = state
        dataloader_state = self._export_dataloader_state()
        return TrainerState(
            step_context=asdict(runtime.state.step_context) if runtime.state.step_context is not None else None,
            consumed_tokens=None if dataloader_state is None else int(dataloader_state["consumed_tokens"]),
            dataloader=dataloader_state,
            rng=rng_state,
            plugin_states=plugin_states if plugin_states else None,
        )

    def import_trainer_state(self, state: TrainerState) -> None:
        if self._runtime is None:
            raise RuntimeError("StateManager is not bound to RuntimeCore")
        runtime = self._runtime
        from runtime.core import StepContext

        runtime.state.step_context = StepContext(**state.step_context) if state.step_context is not None else StepContext()
        torch.set_rng_state(state.rng.cpu)
        if state.rng.cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state.rng.cuda)
        if self._dataloader is not None and state.dataloader is not None:
            self._dataloader.load_state_dict(state.dataloader)
        if state.plugin_states is not None:
            for plugin in runtime.plugins:
                plugin_state = state.plugin_states.get(plugin.id.value)
                if isinstance(plugin_state, dict):
                    plugin.import_plugin_state(plugin_state)

    def _export_dataloader_state(self) -> dict[str, Any] | None:
        if self._dataloader is None:
            return None
        state = self._dataloader.state_dict()
        if is_dataclass(state) and not isinstance(state, type):
            return asdict(state)
        if isinstance(state, dict):
            return dict(state)
        raise TypeError(f"dataloader state must be a dataclass or dict, got {type(state)!r}")
