from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from parallel.specs import TpSpComm, TpSpShardAxis, TpSpShardRule
from runtime.layers.functional import (
    all_gather,
    row_parallel_reduce_scatter_async,
    sequence_parallel_grouped_linear,
)
from runtime.layers.linear import ColumnParallelLinear, RowParallelLinear
from runtime.plugin import PluginId, TpSpParallelizableModule
from runtime.plugins.tp import TensorParallelTransformPlugin
from runtime.types import SetupPhase


class _FusedColumnParallelLinear(ColumnParallelLinear):
    """Column-parallel projection whose output can be produced by its TP/SP parent."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._precomputed_output: torch.Tensor | None = None

    def set_precomputed_output(self, output: torch.Tensor) -> None:
        if self._precomputed_output is not None:
            raise RuntimeError("unconsumed TP/SP projection output")
        self._precomputed_output = output

    def has_precomputed_output(self) -> bool:
        return self._precomputed_output is not None

    def clear_precomputed_output(self) -> None:
        self._precomputed_output = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._precomputed_output is None:
            return super().forward(input)
        output = self._precomputed_output
        self._precomputed_output = None
        if self.bias is not None:
            output = output + self.bias
        return output


class _FusedRowParallelLinear(RowParallelLinear):
    def __init__(self, *args, native_comm_overlap: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.native_comm_overlap = native_comm_overlap

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.comm != "reduce_scatter" or self.world_size <= 1:
            return super().forward(input)
        return row_parallel_reduce_scatter_async(
            input,
            self.weight,
            self.bias,
            self.tp_group,
            alloc_key=f"tp_sp.row_parallel_linear.{id(self)}",
            native_comm_overlap=self.native_comm_overlap,
        )


@dataclass(frozen=True)
class _ProjectionGroup:
    module_path: str
    module: nn.Module
    linears: tuple[_FusedColumnParallelLinear, ...]


class TpSpPlugin(TensorParallelTransformPlugin):
    """Joint tensor/sequence-parallel transform with fused communication plans."""

    def __init__(self, *, native_comm_overlap: bool = False) -> None:
        super().__init__(
            plugin_id=PluginId.TP_SP,
            name="tensor_sequence_parallel",
        )
        self.native_comm_overlap = native_comm_overlap
        self._projection_groups: list[_ProjectionGroup] = []

    @property
    def sp_group(self) -> dist.ProcessGroup:
        group = self.tp_group
        if group is None:
            raise ValueError("TpSpPlugin requires a TP process group")
        return group

    @property
    def rank(self) -> int:
        return dist.get_rank(self.sp_group)

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self.sp_group)

    def on_setup_phase(self, phase: SetupPhase, model: nn.Module) -> nn.Module:
        model = super().on_setup_phase(phase, model)
        if phase == SetupPhase.TRANSFORM:
            self._build_projection_groups(model)
        elif phase == SetupPhase.FINALIZE:
            self._register_sequence_hooks(model)
        return model

    def _make_column_parallel_linear(
        self,
        module: nn.Linear,
        rule: TpSpShardRule,
    ) -> ColumnParallelLinear:
        return _FusedColumnParallelLinear(
            module.in_features,
            module.out_features,
            self.tp_group,
            bias=module.bias is not None,
            gather_output=(rule.post_comm == TpSpComm.ALL_GATHER),
            init=False,
        )

    def _make_row_parallel_linear(
        self,
        module: nn.Linear,
        rule: TpSpShardRule,
    ) -> RowParallelLinear:
        return _FusedRowParallelLinear(
            module.in_features,
            module.out_features,
            self.tp_group,
            rule.post_comm,
            rule.comm_dim,
            bias=module.bias is not None,
            init=False,
            native_comm_overlap=self.native_comm_overlap,
        )

    def _build_projection_groups(self, model: nn.Module) -> None:
        self._projection_groups.clear()
        if not isinstance(model, TpSpParallelizableModule):
            return
        assert self.runtime is not None
        for rule in model.tpsp_parallelize_spec().rules:
            if self.runtime.is_module_path_omitted(rule.module_path):
                continue
            if (
                rule.shard_axis != TpSpShardAxis.SEQUENCE
                or rule.pre_comm != TpSpComm.ALL_GATHER
            ):
                continue
            module = model.get_submodule(rule.module_path)
            linears = tuple(
                child
                for child in module.children()
                if isinstance(child, _FusedColumnParallelLinear)
            )
            if linears:
                self._projection_groups.append(
                    _ProjectionGroup(rule.module_path, module, linears)
                )

    def param_materialization_groups(
        self,
    ) -> tuple[tuple[nn.Module, tuple[nn.Parameter, ...]], ...]:
        return tuple(
            (
                group.module,
                tuple(
                    param
                    for linear in group.linears
                    for param in linear.parameters(recurse=True)
                ),
            )
            for group in self._projection_groups
        )

    def _register_sequence_hooks(self, model: nn.Module) -> None:
        if not isinstance(model, TpSpParallelizableModule):
            return
        assert self.runtime is not None
        groups_by_module = {
            group.module: group
            for group in self._projection_groups
        }
        for rule in model.tpsp_parallelize_spec().rules:
            if self.runtime.is_module_path_omitted(rule.module_path):
                continue
            if rule.shard_axis != TpSpShardAxis.SEQUENCE:
                continue
            module = model.get_submodule(rule.module_path)
            if rule.pre_comm == TpSpComm.ALL_GATHER:
                group = groups_by_module.get(module)
                module.register_forward_pre_hook(
                    self._make_all_gather_hook(rule, group)
                )
                if group is not None:
                    module.register_forward_hook(self._make_projection_cleanup_hook(group))
            elif rule.post_comm == TpSpComm.SCATTER:
                module.register_forward_hook(self._make_scatter_hook(rule.comm_dim))
            else:
                raise NotImplementedError(
                    "TpSpPlugin supports sequence rules with "
                    f"pre_comm='all_gather' or post_comm='scatter', got "
                    f"pre_comm={rule.pre_comm!r}, post_comm={rule.post_comm!r}"
                )

    def _make_all_gather_hook(
        self,
        rule: TpSpShardRule,
        group: _ProjectionGroup | None,
    ):
        def hook(module, input):
            assert self.runtime is not None
            x, *args = input
            microbatch_idx = self.runtime.state.step_context.pp_cur_microbatch_idx
            if group is not None and self._weights_materialized(group):
                x, outputs = sequence_parallel_grouped_linear(
                    x,
                    tuple(linear.weight for linear in group.linears),
                    self.sp_group,
                    alloc_key=(
                        f"tp_sp.{group.module_path}.mb{microbatch_idx}"
                        ".grouped_linear"
                    ),
                    native_comm_overlap=self.native_comm_overlap,
                )
                for linear, output in zip(group.linears, outputs, strict=True):
                    linear.set_precomputed_output(output)
                return (x, *args)
            x = all_gather(
                x,
                self.sp_group,
                rule.comm_dim,
                alloc_key=(
                    f"tp_sp.{rule.module_path}.mb{microbatch_idx}.all_gather"
                ),
                backward_reduce_op=(
                    dist.ReduceOp.SUM if group is not None else None
                ),
            )
            return (x, *args)

        return hook

    @staticmethod
    def _weights_materialized(group: _ProjectionGroup) -> bool:
        return all(
            linear.weight.ndim == 2
            and tuple(linear.weight.shape)
            == (linear.out_features_per_shard, linear.in_features)
            for linear in group.linears
        )

    @staticmethod
    def _make_projection_cleanup_hook(group: _ProjectionGroup):
        def hook(module, input, output):
            unconsumed = [
                linear
                for linear in group.linears
                if linear.has_precomputed_output()
            ]
            for linear in unconsumed:
                linear.clear_precomputed_output()
            if unconsumed:
                raise RuntimeError(
                    f"{type(module).__name__} did not consume every fused TP/SP "
                    "projection output"
                )
            return output

        return hook

    def _make_scatter_hook(self, comm_dim: int):
        def hook(module, input, output):
            per_rank_dim = output.size(comm_dim) // self.world_size
            return torch.narrow(
                output,
                dim=comm_dim,
                start=self.rank * per_rank_dim,
                length=per_rank_dim,
            )

        return hook
