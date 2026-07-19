from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from parallel.specs import TpSpShardAxis
from runtime.mesh import MeshAxis
from runtime.layers.distributed_rmsnorm import DistributedRMSNorm
from runtime.plugin import PluginId, RuntimePlugin, TpSpParallelizableModule
from runtime.types import ParamRole, SetupPhase
from runtime.layers.linear import ColumnParallelLinear, RowParallelLinear


@dataclass
class _LinearInitSpec:
    module_path: str
    shard_axis: TpSpShardAxis
    weight: torch.Tensor
    bias: torch.Tensor | None


@dataclass
class _NormInitSpec:
    module_path: str
    weight: torch.Tensor


class TensorParallelPlugin(RuntimePlugin):
    """Draft TP plugin that cooperates with RuntimeCore instead of driving execution."""

    def __init__(self):
        super().__init__(id=PluginId.TP, name="tensor_parallel")
        self._param_shard_axis: dict[str, TpSpShardAxis] = {}
        self._logical_shapes: dict[str, tuple[int, ...]] = {}
        self._tp_replicated_params: set[str] = set()
        self._pending_linear_inits: list[_LinearInitSpec] = []
        self._pending_norm_inits: list[_NormInitSpec] = []

    @property
    def tp_group(self):
        assert self.runtime is not None
        return self.runtime.get_group(MeshAxis.TP)

    def on_setup_phase(self, phase: SetupPhase, model: nn.Module) -> nn.Module:
        if phase == SetupPhase.TRANSFORM:
            return self._transform_model(model)
        if phase == SetupPhase.MATERIALIZE:
            self._materialize_shards(model)
        return model

    def _transform_model(self, model: nn.Module) -> nn.Module:
        if not isinstance(model, TpSpParallelizableModule):
            return model
        assert self.runtime is not None
        spec = model.tpsp_parallelize_spec()
        for rule in spec.rules:
            if self.runtime.is_module_path_omitted(rule.module_path):
                continue
            module = model.get_submodule(rule.module_path)
            if isinstance(module, nn.Linear):
                if rule.shard_axis == TpSpShardAxis.PARAM_OUT:
                    self._record_linear_rule(rule.module_path, module, rule.shard_axis)
                    self._pending_linear_inits.append(
                        _LinearInitSpec(
                            module_path=rule.module_path,
                            shard_axis=rule.shard_axis,
                            weight=module.weight.detach(),
                            bias=None if module.bias is None else module.bias.detach(),
                        )
                    )
                    new_col = ColumnParallelLinear(
                        module.in_features,
                        module.out_features,
                        self.tp_group,
                        bias=module.bias is not None,
                        gather_output=(rule.post_comm == "all_gather"),
                        init=False,
                    )
                    model.set_submodule(rule.module_path, new_col)
                elif rule.shard_axis == TpSpShardAxis.PARAM_IN:
                    self._record_linear_rule(rule.module_path, module, rule.shard_axis)
                    self._pending_linear_inits.append(
                        _LinearInitSpec(
                            module_path=rule.module_path,
                            shard_axis=rule.shard_axis,
                            weight=module.weight.detach(),
                            bias=None if module.bias is None else module.bias.detach(),
                        )
                    )
                    new_row = RowParallelLinear(
                        module.in_features,
                        module.out_features,
                        self.tp_group,
                        rule.post_comm,
                        rule.comm_dim,
                        bias=module.bias is not None,
                        init=False,
                    )
                    model.set_submodule(rule.module_path, new_row)
                continue
            if isinstance(module, DistributedRMSNorm):
                if rule.shard_axis != TpSpShardAxis.PARAM_OUT:
                    raise ValueError(
                        f"DistributedRMSNorm only supports PARAM_OUT sharding, got {rule.shard_axis} for {rule.module_path}"
                    )
                self._record_norm_rule(rule.module_path, module, rule.shard_axis)
                self._pending_norm_inits.append(
                    _NormInitSpec(module_path=rule.module_path, weight=module.weight.detach())
                )
                world_size = torch.distributed.get_world_size(self.tp_group)
                hidden_size = module.logical_hidden_size // world_size
                model.set_submodule(
                    rule.module_path,
                    DistributedRMSNorm(
                        hidden_size,
                        float(module.eps),
                        tp_group=self.tp_group,
                        logical_hidden_size=module.logical_hidden_size,
                        init_weight=False,
                    ),
                )
        self.runtime.add_module_replacement(nn.Linear, ColumnParallelLinear)
        self.runtime.add_module_replacement(nn.Linear, RowParallelLinear)
        self.runtime.add_module_replacement(DistributedRMSNorm, DistributedRMSNorm)
        return model

    def _materialize_shards(self, model: nn.Module) -> None:
        for spec in self._pending_linear_inits:
            module = model.get_submodule(spec.module_path)
            if isinstance(module, ColumnParallelLinear):
                self._init_column_parallel_linear(module, spec)
            elif isinstance(module, RowParallelLinear):
                self._init_row_parallel_linear(module, spec)
            else:
                raise TypeError(
                    f"expected TP linear module at path={spec.module_path}, got {type(module).__name__}"
                )
        self._pending_linear_inits.clear()

        for spec in self._pending_norm_inits:
            module = model.get_submodule(spec.module_path)
            if not isinstance(module, DistributedRMSNorm):
                raise TypeError(
                    f"expected DistributedRMSNorm at path={spec.module_path}, got {type(module).__name__}"
                )
            self._init_distributed_rmsnorm(module, spec.weight)
        self._pending_norm_inits.clear()

    def _init_column_parallel_linear(self, module: ColumnParallelLinear, spec: _LinearInitSpec) -> None:
        rank = module.rank
        dim_per_shard = spec.weight.size(0) // module.world_size
        weight_shard = spec.weight[rank * dim_per_shard : (rank + 1) * dim_per_shard]
        with torch.no_grad():
            module.weight.copy_(weight_shard.to(device=module.weight.device, dtype=module.weight.dtype))
            if spec.bias is not None and module.bias is not None:
                bias_shard = spec.bias[rank * dim_per_shard : (rank + 1) * dim_per_shard]
                module.bias.copy_(bias_shard.to(device=module.bias.device, dtype=module.bias.dtype))

    def _init_row_parallel_linear(self, module: RowParallelLinear, spec: _LinearInitSpec) -> None:
        rank = module.rank
        dim_per_shard = spec.weight.size(1) // module.world_size
        start, end = rank * dim_per_shard, (rank + 1) * dim_per_shard
        with torch.no_grad():
            module.weight.copy_(spec.weight[:, start:end].to(device=module.weight.device, dtype=module.weight.dtype))
            if spec.bias is not None and module.bias is not None:
                module.bias.copy_(spec.bias.to(device=module.bias.device, dtype=module.bias.dtype))

    def _init_distributed_rmsnorm(self, module: DistributedRMSNorm, full_weight: torch.Tensor) -> None:
        shard = full_weight.numel() // module.world_size
        start = module.rank * shard
        end = (module.rank + 1) * shard
        with torch.no_grad():
            module.weight.copy_(full_weight[start:end].to(device=module.weight.device, dtype=module.weight.dtype))

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

    def _record_norm_rule(self, module_path: str, module: DistributedRMSNorm, shard_axis: TpSpShardAxis) -> None:
        weight_name = f"{module_path}.weight"
        self._param_shard_axis[weight_name] = shard_axis
        self._logical_shapes[weight_name] = (module.logical_hidden_size,)

    def annotate_param_metadata(self) -> None:
        assert self.runtime is not None
        for fq_name, logical_shape in self._logical_shapes.items():
            param = self.runtime.state_manager.get_model_tensor(fq_name)
            self.runtime.state_manager.update_model_state(
                fq_name,
                logical_names=[fq_name],
                logical_shapes=[logical_shape],
                physical_shape=tuple(param.shape),
                dtype=str(param.dtype),
            )
        if self.runtime.mesh.tp <= 1:
            return
        for fq_name in self._param_shard_axis:
            attrs = self.runtime.state_manager.params[fq_name].attrs
            self.runtime.state_manager.update_model_state(
                fq_name,
                sharded_axes=attrs.sharded_axes | {MeshAxis.TP},
            )
        for fq_name, entry in self.runtime.state_manager.params.items():
            if fq_name in self._param_shard_axis:
                continue
            if fq_name not in self._tp_replicated_params and self.runtime.get_param_role(entry.param) != ParamRole.SHARED:
                continue
            attrs = entry.attrs
            self.runtime.state_manager.update_model_state(
                fq_name,
                replicated_axes=attrs.replicated_axes | {MeshAxis.TP},
            )
