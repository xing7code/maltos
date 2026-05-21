import math
from absl import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from enum import Enum

from train_system.runtime.plugin import BaseParallelPlugin, ParallelizableModule
from train_system.runtime.functional import all_gather, all_reduce, reduce_scatter


class ColumnParallelLinear(nn.Module):

    def __init__(self, in_features, out_features, tp_group, bias=True, gather_output=True, init=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        self.tp_group = tp_group
        self.rank = dist.get_rank(tp_group)
        self.world_size = dist.get_world_size(tp_group)

        assert self.out_features % self.world_size == 0, "Out features must be divisible by world size"
        self.out_features_per_shard = self.out_features // self.world_size
        self.weight = nn.Parameter(torch.empty(self.out_features_per_shard, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_shard))
        else:
            self.bias = self.register_parameter("bias", None)
        if init:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1/math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, linear_module, tp_group, gather_output=True):
        col_linear = cls(
            linear_module.in_features,
            linear_module.out_features,
            tp_group,
            bias=linear_module.bias is not None,
            gather_output=gather_output,
            init=False,
        ).to(linear_module.weight.device, dtype=linear_module.weight.dtype)
        rank = col_linear.rank
        world_size = col_linear.world_size
        with torch.no_grad():
            full_weight = linear_module.weight
            dim_per_shard = full_weight.size(0)//world_size
            col_linear.weight.copy_(full_weight[rank*dim_per_shard : (rank+1)*dim_per_shard])
            if linear_module.bias is not None:
                col_linear.bias.copy_(linear_module.bias[rank*dim_per_shard : (rank+1)*dim_per_shard])

        return col_linear

    def forward(self, input):
        output = F.linear(input, self.weight, self.bias)
        if self.gather_output and self.world_size>1:
            output = all_gather(output, self.tp_group, -1)
        return output


class RowParallelLinear(nn.Module):

    def __init__(self, in_features, out_features, tp_group, comm="none", comm_dim=0, bias=True, init=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.tp_group = tp_group
        self.comm = comm
        self.comm_dim = comm_dim
        self.rank = dist.get_rank(tp_group)
        self.world_size = dist.get_world_size(tp_group)

        assert self.in_features % self.world_size == 0, "In features must be divisible by world size"
        self.in_features_per_shard = self.in_features // self.world_size
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features_per_shard))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.bias = self.register_parameter("bias", None)
        if init:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1/math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, linear_module, tp_group, comm, comm_dim=0):
        row_linear = cls(
            linear_module.in_features,
            linear_module.out_features,
            tp_group,
            comm,
            comm_dim,
            bias=linear_module.bias is not None,
            init=False
        ).to(linear_module.weight.device, dtype=linear_module.weight.dtype)
        rank = row_linear.rank
        world_size = row_linear.world_size
        with torch.no_grad():
            dim_per_shard = linear_module.in_features // world_size
            start, end = rank*dim_per_shard, (rank+1)*dim_per_shard
            row_linear.weight.copy_(linear_module.weight[:, start:end])
            if row_linear.bias is not None:
                row_linear.bias.copy_(linear_module.bias)
        return row_linear

    def forward(self, input):
        output = F.linear(input, self.weight, None)
        if self.comm == "all_reduce":
            output = all_reduce(output, self.tp_group, dist.ReduceOp.SUM)
        elif self.comm == "reduce_scatter":
            output = reduce_scatter(output, self.tp_group, 1)
        if self.bias is not None:
            output = output + self.bias
        return output


class TpPlugin(BaseParallelPlugin):

    _COL_POST_COMM = {"all_gather", "none"}
    _ROW_POST_COMM = {"all_reduce", "reduce_scatter", "none"}
    _SUPPORTED_PRE_COMM = {"none"}

    def __init__(self, tp_group: dist.ProcessGroup):
        self.tp_group = tp_group

    def setup_model(self, model: nn.Module) -> nn.Module:
        if not isinstance(model, ParallelizableModule):
            logging.warn(f"model {model._get_name()} did not impl parallelize_spec, ignoring TpPlugin!")
            return model
        spec = model.parallelize_spec()
        for r in spec.rules:
            mod = model.get_submodule(r.module_path)
            if r.shard_style not in ("col", "row"):
                continue
            if r.pre_comm not in self._SUPPORTED_PRE_COMM:
                raise NotImplementedError(
                    f"TpPlugin does not support pre_comm={r.pre_comm!r} yet on module {r.module_path}."
                )
            if not isinstance(mod, nn.Linear):
                print(f"invalid shard_style {r.shard_style} on module {mod.__class__}, ignoring")
                continue
            if r.shard_style == "col":
                if r.post_comm not in self._COL_POST_COMM:
                    raise ValueError(
                        f"Invalid post_comm={r.post_comm!r} for col shard on {r.module_path}. "
                        "Expected one of {'all_gather', 'none', None}."
                    )
                model.set_submodule(
                    r.module_path,
                    ColumnParallelLinear.from_linear(mod, self.tp_group, gather_output=(r.post_comm=="all_gather"))
                )
            elif r.shard_style == "row":
                if r.post_comm not in self._ROW_POST_COMM:
                    raise ValueError(
                        f"Invalid post_comm={r.post_comm!r} for row shard on {r.module_path}. "
                        "Expected one of {'all_reduce', 'reduce_scatter', 'none', None}."
                    )
                model.set_submodule(
                    r.module_path,
                    RowParallelLinear.from_linear(mod, self.tp_group, r.post_comm, r.comm_dim)
                )
        return model
