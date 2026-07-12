from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime.core import RuntimeCore

import torch
import torch.nn as nn
import torch.distributed as dist

from parallel.specs import TpSpComm, TpSpShardAxis, TpSpShardRule
from runtime.layers.functional import all_gather
from runtime.mesh import MeshAxis
from runtime.plugin import PluginId, RuntimePlugin, TpSpParallelizableModule


class SequenceParallelPlugin(RuntimePlugin):
    """Sequence parallel hooks for the RuntimeCore plugin path.

    SP composes with TP in this minimal runtime: TP rewrites the transformer
    linears first, then SP registers activation layout hooks around modules
    annotated with ``shard_axis=SEQUENCE``.
    """

    def __init__(self) -> None:
        super().__init__(
            id=PluginId.SP,
            name="sequence_parallel",
            requires={PluginId.TP},
            runs_after={PluginId.TP},
        )

    @property
    def sp_group(self) -> dist.ProcessGroup:
        assert self.runtime is not None
        group = self.runtime.get_group(MeshAxis.TP)
        if group is None:
            raise ValueError("SequenceParallelPlugin requires a TP process group")
        return group

    @property
    def rank(self) -> int:
        return dist.get_rank(self.sp_group)

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self.sp_group)

    def transform_model(self, model: nn.Module) -> nn.Module:
        if not isinstance(model, TpSpParallelizableModule):
            return model
        assert self.runtime is not None

        for rule in model.tpsp_parallelize_spec().rules:
            if self.runtime.is_module_path_omitted(rule.module_path):
                continue
            if rule.shard_axis != TpSpShardAxis.SEQUENCE:
                continue
            module = model.get_submodule(rule.module_path)
            self._register_sequence_hook(module, rule)
        return model

    def _register_sequence_hook(self, module: nn.Module, rule: TpSpShardRule) -> None:
        if rule.pre_comm == TpSpComm.ALL_GATHER:
            module.register_forward_pre_hook(self._make_all_gather_hook(rule.comm_dim, rule.module_path))
        elif rule.post_comm == TpSpComm.SCATTER:
            module.register_forward_hook(self._make_scatter_hook(rule.comm_dim))
        else:
            raise NotImplementedError(
                "SequenceParallelPlugin supports sequence rules with "
                f"pre_comm='all_gather' or post_comm='scatter', got "
                f"pre_comm={rule.pre_comm!r}, post_comm={rule.post_comm!r}"
            )

    def _make_all_gather_hook(self, comm_dim: int, module_path: str):
        def hook(module, input):
            x, *args = input
            x = all_gather(
                x,
                self.sp_group,
                comm_dim,
                alloc_key=f"sp.{module_path}.all_gather",
            )
            return (x, *args)

        return hook

    def _make_scatter_hook(self, comm_dim: int):
        def hook(module, input, output):
            per_rank_dim = output.size(comm_dim) // self.world_size
            return torch.narrow(output, dim=comm_dim, start=self.rank * per_rank_dim, length=per_rank_dim)

        return hook
