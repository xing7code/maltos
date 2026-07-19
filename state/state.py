from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from runtime.mesh import MeshAxis
from runtime.types import StepContext

if TYPE_CHECKING:
    from data import StatefulDataLoaderProtocol
    from runtime.core import RuntimeCore


@dataclass
class RuntimeParamAttrs:
    """Runtime-only metadata for one checkpointed model-state tensor."""

    sharded_axes: set[MeshAxis] = field(default_factory=set)
    replicated_axes: set[MeshAxis] = field(default_factory=set)


@dataclass
class ModelStateMeta:
    state_key: str
    logical_names: list[str]
    logical_shapes: list[tuple[int, ...]]
    physical_shape: tuple[int, ...]
    dtype: str
    source_rank: int | None = None


@dataclass
class ParamEntry:
    meta: ModelStateMeta
    param: nn.Parameter
    attrs: RuntimeParamAttrs = field(default_factory=RuntimeParamAttrs)


@dataclass
class BufferEntry:
    meta: ModelStateMeta
    tensor: torch.Tensor


@dataclass(frozen=True)
class RngState:
    cpu: torch.Tensor
    cuda: list[torch.Tensor] | None = None


@dataclass(frozen=True)
class TrainerState:
    rng: RngState
    step_context: dict[str, Any] | None = None
    dataloader: dict[str, Any] | None = None
    plugin_states: dict[str, dict[str, Any]] | None = None


@dataclass(frozen=True)
class OptimizerState:
    state: dict[str, Any]


@dataclass
class StateManager:
    """Runtime-owned index of checkpointable model and training state."""

    params: dict[str, ParamEntry] = field(default_factory=dict)
    buffers: dict[str, BufferEntry] = field(default_factory=dict)
    _param_names_by_id: dict[int, str] = field(default_factory=dict, repr=False)
    _runtime: "RuntimeCore | None" = field(default=None, init=False, repr=False)
    _dataloader: "StatefulDataLoaderProtocol | None" = field(default=None, init=False, repr=False)

    _OPTIMIZER_STATE_PREFIX = "__optimizer_state__."
    _SCHEDULER_STATE_PREFIX = "__scheduler_state__."

    def _optimizer_state_key(self, owner: str) -> str:
        return f"{self._OPTIMIZER_STATE_PREFIX}{owner}"

    def _scheduler_state_key(self, owner: str) -> str:
        return f"{self._SCHEDULER_STATE_PREFIX}{owner}"

    def _resolve_runtime_name(self, fq_name: str) -> str:
        if (
            fq_name in self.params
            or fq_name in self.buffers
        ):
            return fq_name
        with_prefix = f"module.{fq_name}"
        if (
            with_prefix in self.params
            or with_prefix in self.buffers
        ):
            return with_prefix
        if fq_name.startswith("module."):
            without_prefix = fq_name[len("module.") :]
            if (
                without_prefix in self.params
                or without_prefix in self.buffers
            ):
                return without_prefix
        return fq_name

    def register_module(self, model: nn.Module) -> None:
        self.params.clear()
        self.buffers.clear()
        self._param_names_by_id.clear()
        for fq_name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self._param_names_by_id[id(param)] = fq_name
            self.params[fq_name] = ParamEntry(
                meta=ModelStateMeta(
                    state_key=fq_name,
                    logical_names=[fq_name],
                    logical_shapes=[tuple(param.shape)],
                    physical_shape=tuple(param.shape),
                    dtype=str(param.dtype),
                ),
                param=param,
            )
        param_names = set(self.params)
        persistent_buffer_names = set(model.state_dict()) - param_names
        for fq_name, buffer in model.named_buffers():
            if fq_name not in persistent_buffer_names:
                continue
            self.buffers[fq_name] = BufferEntry(
                meta=ModelStateMeta(
                    state_key=fq_name,
                    logical_names=[fq_name],
                    logical_shapes=[tuple(buffer.shape)],
                    physical_shape=tuple(buffer.shape),
                    dtype=str(buffer.dtype),
                ),
                tensor=buffer,
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

    @property
    def param_states(self) -> Mapping[str, ModelStateMeta]:
        return {fq_name: entry.meta for fq_name, entry in self.params.items()}

    @property
    def buffer_states(self) -> Mapping[str, ModelStateMeta]:
        return {fq_name: entry.meta for fq_name, entry in self.buffers.items()}

    def update_model_state(
        self,
        fq_name: str,
        *,
        logical_names: list[str] | None = None,
        logical_shapes: list[tuple[int, ...]] | None = None,
        physical_shape: tuple[int, ...] | None = None,
        dtype: str | None = None,
        source_rank: int | None = None,
        sharded_axes: set[MeshAxis] | None = None,
        replicated_axes: set[MeshAxis] | None = None,
    ) -> None:
        fq_name = self._resolve_runtime_name(fq_name)
        param_entry = self.params.get(fq_name)
        if param_entry is not None:
            meta = param_entry.meta
            attrs = param_entry.attrs
        else:
            buffer_entry = self.buffers.get(fq_name)
            meta = buffer_entry.meta if buffer_entry is not None else None
            attrs = None
        if meta is None:
            raise KeyError(f"model state not found: {fq_name}")
        if logical_names is not None:
            meta.logical_names = logical_names
        if logical_shapes is not None:
            meta.logical_shapes = logical_shapes
        if physical_shape is not None:
            meta.physical_shape = physical_shape
        if dtype is not None:
            meta.dtype = dtype
        if source_rank is not None:
            meta.source_rank = source_rank
        if attrs is not None:
            if sharded_axes is not None:
                attrs.sharded_axes = sharded_axes
            if replicated_axes is not None:
                attrs.replicated_axes = replicated_axes

    def get_model_tensor(self, fq_name: str) -> torch.Tensor:
        fq_name = self._resolve_runtime_name(fq_name)
        if fq_name in self.params:
            return self.params[fq_name].param
        if fq_name in self.buffers:
            return self.buffers[fq_name].tensor
        raise KeyError(f"runtime model tensor not found: {fq_name}")

    def get_param_name(self, param: nn.Parameter) -> str:
        fq_name = self._param_names_by_id.get(id(param))
        if fq_name is None:
            raise KeyError("runtime parameter name not found")
        return fq_name

    def export_optimizer_state(self) -> OptimizerState | None:
        runtime = self.runtime
        optimizer, scheduler = runtime.get_optimizer_and_scheduler()
        if optimizer is None:
            return None
        owner = "runtime" if runtime._optimizer_owner is runtime else runtime._optimizer_owner.id.value
        state: dict[str, Any] = {self._optimizer_state_key(owner): optimizer.state_dict()}
        if scheduler is not None:
            state[self._scheduler_state_key(owner)] = scheduler.state_dict()
        return OptimizerState(state=state)

    def import_optimizer_state(self, state: OptimizerState) -> None:
        runtime = self.runtime
        payload = state.state
        optimizer, scheduler = runtime.get_optimizer_and_scheduler()
        if optimizer is None:
            raise ValueError("no optimizer available to load state into")
        owner = "runtime" if runtime._optimizer_owner is runtime else runtime._optimizer_owner.id.value
        optimizer_state = payload.get(self._optimizer_state_key(owner))
        if optimizer_state is None:
            raise ValueError(f"no optimizer state found for owner={owner}")
        optimizer.load_state_dict(optimizer_state)
        scheduler_state = payload.get(self._scheduler_state_key(owner))
        if scheduler_state is not None and scheduler is not None:
            scheduler.load_state_dict(scheduler_state)

    def export_model_state(self) -> tuple[dict[str, torch.Tensor], list[ModelStateMeta]]:
        return self.runtime._model_state_owner.export_model_state(self)

    def import_model_state(self, model_state: dict[str, torch.Tensor]) -> None:
        self.runtime._model_state_owner.import_model_state(self, model_state)

    def export_trainer_state(self) -> TrainerState:
        runtime = self.runtime
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
            dataloader=dataloader_state,
            rng=rng_state,
            plugin_states=plugin_states if plugin_states else None,
        )

    def import_trainer_state(self, state: TrainerState) -> None:
        runtime = self.runtime

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
