from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime.core import RuntimeCore

import torch
import torch.distributed as dist
import torch.nn as nn

from parallel.specs import TpSpParallelSpec
from runtime.mesh import MeshAxis
from runtime.plugin import PipelineParallelizableModule, PluginId, RuntimePlugin, TpSpParallelizableModule
from runtime.step_runners import PipelineScheduleKind, PipelineStepRunner


class PipelineParallelPlugin(RuntimePlugin):
    def __init__(self, schedule: str = "afab") -> None:
        super().__init__(
            id=PluginId.PP,
            name="pipeline_parallel",
            owns_step_runner=True,
            runs_before={
                PluginId.DP,
                PluginId.ZERO1,
                PluginId.ZERO2,
                PluginId.ZERO3,
                PluginId.TP,
                PluginId.SP,
                PluginId.EP,
                PluginId.PRECISION,
            },
        )
        self.stage_index = 0
        self.stage_count = 1
        self.prev_global_rank: int | None = None
        self.next_global_rank: int | None = None
        self.last_global_rank = 0
        self.hidden_size = 0
        self.pp_group: dist.ProcessGroup | None = None
        self.sequence_parallel_enabled = False
        self.sequence_parallel_world_size = 1
        self.schedule = PipelineScheduleKind(schedule)

    def bind(self, runtime: "RuntimeCore") -> None:
        super().bind(runtime)
        active_plugins = {plugin.id for plugin in runtime.plugins if plugin is not self}
        self.sequence_parallel_enabled = PluginId.SP in active_plugins
        self.sequence_parallel_world_size = runtime.mesh.tp if self.sequence_parallel_enabled else 1

    def transform_model(self, model: nn.Module) -> nn.Module:
        assert self.runtime is not None
        self._validate_runtime_support()
        if not dist.is_initialized():
            raise ValueError("PipelineParallelPlugin requires torch.distributed to be initialized")

        self.pp_group = self.runtime.get_group(MeshAxis.PP)
        global_rank = dist.get_rank()
        _, pp_idx, _, _ = self.runtime.mesh.rank_coordinates(global_rank)
        self.stage_index = pp_idx
        self.stage_count = self.runtime.mesh.pp
        self.prev_global_rank = self._stage_global_rank(pp_idx - 1) if pp_idx > 0 else None
        self.next_global_rank = self._stage_global_rank(pp_idx + 1) if pp_idx + 1 < self.stage_count else None
        self.last_global_rank = self._stage_global_rank(self.stage_count - 1)
        if not isinstance(model, PipelineParallelizableModule):
            raise TypeError(
                "PipelineParallelPlugin requires model.pipeline_parallel_spec(), "
                f"got {type(model).__name__}"
            )
        spec = model.pipeline_parallel_spec()
        self.hidden_size = _infer_hidden_size(model, spec)
        self._partition_model(model, spec, pp_idx, self.stage_count)
        self._filter_tpsp_spec(model, spec, pp_idx, self.stage_count)
        return model

    def build_step_runner(self):
        return PipelineStepRunner(self)

    def _stage_global_rank(self, stage_index: int) -> int:
        assert self.runtime is not None
        global_rank = dist.get_rank()
        dp_idx, _, cp_idx, tp_idx = self.runtime.mesh.rank_coordinates(global_rank)
        return self.runtime.mesh.rank_id(dp=dp_idx, pp=stage_index, cp=cp_idx, tp=tp_idx)

    def _validate_runtime_support(self) -> None:
        assert self.runtime is not None
        mesh = self.runtime.mesh
        if mesh.pp <= 1:
            raise ValueError("PipelineParallelPlugin requires mesh.pp > 1")
        unsupported = set()
        active = {plugin.id for plugin in self.runtime.plugins if plugin is not self}
        overlap = sorted(plugin_id.value for plugin_id in active & unsupported)
        if overlap:
            raise ValueError(f"PipelineParallelPlugin does not yet support plugin combinations: {overlap}")

    def _partition_model(self, model: nn.Module, spec, stage_index: int, stage_count: int) -> None:
        assert self.runtime is not None
        for path in spec.head_layers:
            if stage_index != 0:
                self.runtime.mark_module_path_omitted(path)
                _replace_module_path(model, path, None)
        for path in spec.tail_layers:
            if stage_index != stage_count - 1:
                self.runtime.mark_module_path_omitted(path)
                _replace_module_path(model, path, None)

        for path in spec.pipe_layers:
            module = model.get_submodule(path)
            if not isinstance(module, nn.ModuleList):
                raise TypeError(
                    f"PipelineParallelPlugin expects pipe layer path={path!r} to be nn.ModuleList, "
                    f"got {type(module).__name__}"
                )
            start, end = _layer_range(len(module), stage_index, stage_count)
            for layer_idx in range(len(module)):
                if not start <= layer_idx < end:
                    self.runtime.mark_module_path_omitted(f"{path}.{layer_idx}")
            partitioned = nn.ModuleList(
                [layer if start <= layer_idx < end else _IdentityPipeLayer() for layer_idx, layer in enumerate(module)]
            )
            _replace_module_path(model, path, partitioned)

    def _filter_tpsp_spec(self, model: nn.Module, pp_spec, stage_index: int, stage_count: int) -> None:
        if not isinstance(model, TpSpParallelizableModule):
            return
        tpsp_spec = model.tpsp_parallelize_spec()
        keep_prefixes: list[str] = []
        if stage_index == 0:
            keep_prefixes.extend(pp_spec.head_layers)
        if stage_index == stage_count - 1:
            keep_prefixes.extend(pp_spec.tail_layers)
        for path in pp_spec.pipe_layers:
            module = model.get_submodule(path)
            if not isinstance(module, nn.ModuleList):
                continue
            start, end = _layer_range(len(module), stage_index, stage_count)
            keep_prefixes.extend(f"{path}.{layer_idx}" for layer_idx in range(start, end))

        filtered_rules = [rule for rule in tpsp_spec.rules if _path_matches_any_prefix(rule.module_path, keep_prefixes)]
        filtered_tie_rules = [
            tie_rule
            for tie_rule in tpsp_spec.tie_rules
            if _path_matches_any_prefix(tie_rule[0], keep_prefixes)
            and _path_matches_any_prefix(tie_rule[1], keep_prefixes)
        ]
        filtered_spec = TpSpParallelSpec(rules=filtered_rules, tie_rules=filtered_tie_rules)
        model.tpsp_parallelize_spec = MethodType(lambda _self: filtered_spec, model)


def _layer_range(num_layers: int, stage_index: int, stage_count: int) -> tuple[int, int]:
    base, remainder = divmod(num_layers, stage_count)
    start = stage_index * base + min(stage_index, remainder)
    width = base + (1 if stage_index < remainder else 0)
    return start, start + width


def _path_matches_any_prefix(path: str, prefixes: list[str]) -> bool:
    for prefix in prefixes:
        if path == prefix or path.startswith(prefix + "."):
            return True
    return False


class _IdentityPipeLayer(nn.Module):
    def forward(self, *args, **kwargs):
        if not args:
            raise ValueError("identity pipe layer expects at least one positional argument")
        return args[0]


def _infer_hidden_size(model: nn.Module, spec) -> int:
    for path in spec.head_layers:
        module = model.get_submodule(path)
        if isinstance(module, nn.Embedding):
            return int(module.embedding_dim)
    for path in spec.tail_layers:
        module = model.get_submodule(path)
        weight = getattr(module, "weight", None)
        if torch.is_tensor(weight) and weight.ndim >= 1:
            return int(weight.shape[0])
    raise ValueError("unable to infer PP hidden size from model/spec")


def _replace_module_path(model: nn.Module, path: str, module: nn.Module | None) -> None:
    if "." not in path:
        setattr(model, path, module)
        return
    parent_path, leaf = path.rsplit(".", 1)
    parent = model.get_submodule(parent_path)
    setattr(parent, leaf, module)
