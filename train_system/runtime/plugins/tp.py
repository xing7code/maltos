from __future__ import annotations

import torch.nn as nn

from train_system.runtime.core import RuntimePhase
from train_system.runtime.mesh import MeshAxis
from train_system.runtime.plugin import ParallelizableModule, PluginId, RuntimePlugin
from train_system.runtime.layers.tp import ColumnParallelLinear, RowParallelLinear


class TensorParallelPlugin(RuntimePlugin):
    """Draft TP plugin that cooperates with RuntimeCore instead of driving execution."""

    def __init__(self):
        super().__init__(id=PluginId.TP, name="tensor_parallel")

    @property
    def tp_group(self):
        assert self.runtime is not None
        return self.runtime.get_group(MeshAxis.TP)

    def transform_model(self, model: nn.Module) -> nn.Module:
        if not isinstance(model, ParallelizableModule):
            return model
        spec = model.parallelize_spec()
        for rule in spec.rules:
            module = model.get_submodule(rule.module_path)
            if not isinstance(module, nn.Linear):
                continue
            if rule.shard_style == "col":
                model.set_submodule(
                    rule.module_path,
                    ColumnParallelLinear.from_linear(
                        module,
                        self.tp_group,
                        gather_output=(rule.post_comm == "all_gather"),
                    ),
                )
            elif rule.shard_style == "row":
                model.set_submodule(
                    rule.module_path,
                    RowParallelLinear.from_linear(
                        module,
                        self.tp_group,
                        rule.post_comm,
                        rule.comm_dim,
                    ),
                )
        return model

    def on_phase(self, phase: RuntimePhase) -> None:
        if phase != RuntimePhase.TRANSFORM_MODEL:
            return
        assert self.runtime is not None
        for fq_name, handle in self.runtime.state_registry.items():
            handle.runtime.logical_shape = tuple(handle.param.shape)
            handle.runtime.local_shape = tuple(handle.param.shape)
            handle.runtime.extra = {
                "parallelism": "tp",
                "tp_world_size": 1 if self.tp_group is None else self.runtime.mesh.tp,
                "note": "After module replacement, this metadata can be refined per shard rule.",
            }
