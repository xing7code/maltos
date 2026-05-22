from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch.nn as nn

@dataclass
class RuntimeParamStatus:
    """Runtime-only mutable state for one parameter reference."""
    
    is_materialized: bool = True
    is_gathered: bool = False


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


@dataclass
class OptimizerState:
    state: dict[str, Any]


@dataclass
class TrainerState:
    step: int
    consumed_tokens: int | None = None
    dataloader: dict[str, Any] | None = None


StateContent = ParamState | OptimizerState | TrainerState


@dataclass
class StateShard:
    key: str
    content: StateContent
    annotations: dict[str, object] = field(default_factory=dict)


@dataclass
class RegisteredStateHandler:
    name: str
    save: Callable[[], Any]
    load: Callable[[Any], None]


@dataclass
class StateRegistry:
    """Runtime-owned index of logical training state.

    The current implementation only tracks parameters, but the registry is the
    place where ZeRO/FSDP, optimizer sharding, checkpointing, and profilers will
    agree on logical parameter names, local shards, materialization state, and
    eventually gradient and optimizer-state handles.
    """

    states: dict[str, RegisteredStateHandler] = field(default_factory=dict)
    shards: dict[str, StateShard] = field(default_factory=dict)
    _runtime_params: dict[str, nn.Parameter] = field(default_factory=dict, repr=False)
    _runtime_status: dict[str, RuntimeParamStatus] = field(default_factory=dict, repr=False)

    def register_module(self, model: nn.Module) -> None:
        self.shards.clear()
        self._runtime_params.clear()
        self._runtime_status.clear()
        for fq_name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self._runtime_params[fq_name] = param
            self._runtime_status[fq_name] = RuntimeParamStatus(
                is_materialized=True,
                is_gathered=False,
            )
            self.shards[fq_name] = StateShard(
                key=fq_name,
                content=ParamState(
                    state_key=fq_name,
                    logical_names=[fq_name],
                    logical_shapes=[tuple(param.shape)],
                    physical_shape=tuple(param.shape),
                    dtype=str(param.dtype),
                ),
            )

    def register_state(
        self,
        name: str,
        save: Callable[[], Any],
        load: Callable[[Any], None],
    ) -> None:
        if name in self.states:
            raise ValueError(f"duplicate state registration: {name}")
        self.states[name] = RegisteredStateHandler(name=name, save=save, load=load)

    def dump_state(self) -> dict[str, Any]:
        return {name: entry.save() for name, entry in self.states.items()}

    def load_state(self, state: dict[str, Any], strict: bool = True) -> None:
        unknown = set(state) - set(self.states)
        if strict and unknown:
            unknown_names = sorted(unknown)
            raise ValueError(f"unknown registry states in checkpoint: {unknown_names}")
        for name, value in state.items():
            entry = self.states.get(name)
            if entry is not None:
                entry.load(value)

    def update_param_shard(
        self,
        fq_name: str,
        *,
        logical_names: list[str] | None = None,
        logical_shapes: list[tuple[int, ...]] | None = None,
        physical_shape: tuple[int, ...] | None = None,
        dtype: str | None = None,
    ) -> None:
        shard = self.shards.get(fq_name)
        if shard is None or not isinstance(shard.content, ParamState):
            raise KeyError(f"param shard not found: {fq_name}")
        if logical_names is not None:
            shard.content.logical_names = logical_names
        if logical_shapes is not None:
            shard.content.logical_shapes = logical_shapes
        if physical_shape is not None:
            shard.content.physical_shape = physical_shape
        if dtype is not None:
            shard.content.dtype = dtype

    def get_param_state(self, fq_name: str) -> ParamState:
        shard = self.shards.get(fq_name)
        if shard is None or not isinstance(shard.content, ParamState):
            raise KeyError(f"param shard not found: {fq_name}")
        return shard.content

    def get_param_tensor(self, fq_name: str) -> nn.Parameter:
        if fq_name not in self._runtime_params:
            raise KeyError(f"runtime parameter not found: {fq_name}")
        return self._runtime_params[fq_name]

    def get_param_status(self, fq_name: str) -> RuntimeParamStatus:
        if fq_name not in self._runtime_status:
            raise KeyError(f"runtime status not found: {fq_name}")
        return self._runtime_status[fq_name]

    def iter_param_states(self):
        for key, shard in self.shards.items():
            if isinstance(shard.content, ParamState):
                yield key, shard.content
