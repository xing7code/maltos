from __future__ import annotations

import torch.nn as nn

from parallel.specs import TpSpShardAxis
from runtime.mesh import MeshAxis
from runtime.plugin import PluginId, RuntimePlugin, TpSpParallelizableModule
from runtime.types import ParamRole, RuntimePhase
from runtime.layers.linear import ColumnParallelLinear, RowParallelLinear


class TensorParallelPlugin(RuntimePlugin):
    """Draft TP plugin that cooperates with RuntimeCore instead of driving execution."""

    def __init__(self):
        super().__init__(id=PluginId.TP, name="tensor_parallel")
        self._param_shard_axis: dict[str, TpSpShardAxis] = {}
        self._logical_shapes: dict[str, tuple[int, ...]] = {}
        self._tp_replicated_params: set[str] = set()

    @property
    def tp_group(self):
        assert self.runtime is not None
        return self.runtime.get_group(MeshAxis.TP)

    def transform_model(self, model: nn.Module) -> nn.Module:
        if not isinstance(model, TpSpParallelizableModule):
            return model
        assert self.runtime is not None
        spec = model.tpsp_parallelize_spec()
        for rule in spec.rules:
            if self.runtime.is_module_path_omitted(rule.module_path):
                continue
            module = model.get_submodule(rule.module_path)
            if not isinstance(module, nn.Linear):
                continue
            if rule.shard_axis == TpSpShardAxis.PARAM_OUT:
                self._record_linear_rule(rule.module_path, module, rule.shard_axis)
                new_col = ColumnParallelLinear.from_linear(
                    module,
                    self.tp_group,
                    gather_output=(rule.post_comm == "all_gather"),
                )
                model.set_submodule(rule.module_path, new_col)
            elif rule.shard_axis == TpSpShardAxis.PARAM_IN:
                self._record_linear_rule(rule.module_path, module, rule.shard_axis)
                new_row = RowParallelLinear.from_linear(
                    module,
                    self.tp_group,
                    rule.post_comm,
                    rule.comm_dim,
                )
                model.set_submodule(rule.module_path, new_row)
        self.runtime.add_module_replacement(nn.Linear, ColumnParallelLinear)
        self.runtime.add_module_replacement(nn.Linear, RowParallelLinear)
        return model

    def _record_linear_rule(self, module_path: str, module: nn.Linear, shard_axis: TpSpShardAxis) -> None:
        weight_name = f"{module_path}.weight"
        self._param_shard_axis[weight_name] = shard_axis
        self._logical_shapes[weight_name] = tuple(module.weight.shape)
        if module.bias is not None:
            bias_name = f"{module_path}.bias"
            self._logical_shapes[bias_name] = tuple(module.bias.shape)
            if shard_axis == TpSpShardAxis.PARAM_OUT:
                self._param_shard_axis[bias_name] = shard_axis
            else:
                self._tp_replicated_params.add(bias_name)

    def annotate_param_metadata(self) -> None:
        assert self.runtime is not None
        for fq_name, param_state in self.runtime.state_manager.param_states.items():
            param = self.runtime.state_manager.get_param_tensor(fq_name)
            logical_shape = self._logical_shapes.get(fq_name, tuple(param.shape))
            local_shape = tuple(param.shape)
            self.runtime.state_manager.update_param_state(
                fq_name,
                logical_names=[fq_name],
                logical_shapes=[logical_shape],
                physical_shape=local_shape,
                dtype=str(param.dtype),
            )
            if self.runtime.mesh.tp <= 1:
                continue
            if fq_name in self._param_shard_axis:
                attrs = self.runtime.state_manager.get_param_attrs(fq_name)
                self.runtime.state_manager.update_param_state(
                    fq_name,
                    sharded_axes=attrs.sharded_axes | {MeshAxis.TP},
                )
                continue
            if fq_name in self._tp_replicated_params or self.runtime.get_param_role(param) == ParamRole.SHARED:
                attrs = self.runtime.state_manager.get_param_attrs(fq_name)
                self.runtime.state_manager.update_param_state(
                    fq_name,
                    replicated_axes=attrs.replicated_axes | {MeshAxis.TP},
                )
