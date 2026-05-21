from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from train_system.runtime.layers.functional import all_gather, all_reduce, reduce_scatter


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_group, bias=True, gather_output=True, init=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        self.tp_group = tp_group
        self.rank = dist.get_rank(tp_group)
        self.world_size = dist.get_world_size(tp_group)

        if self.out_features % self.world_size != 0:
            raise ValueError("out_features must be divisible by TP world size")
        self.out_features_per_shard = self.out_features // self.world_size
        self.weight = nn.Parameter(torch.empty(self.out_features_per_shard, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_shard))
        else:
            self.register_parameter("bias", None)
        if init:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
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
            dim_per_shard = full_weight.size(0) // world_size
            col_linear.weight.copy_(full_weight[rank * dim_per_shard : (rank + 1) * dim_per_shard])
            if linear_module.bias is not None:
                col_linear.bias.copy_(linear_module.bias[rank * dim_per_shard : (rank + 1) * dim_per_shard])
        return col_linear

    def forward(self, input):
        output = F.linear(input, self.weight, self.bias)
        if self.gather_output and self.world_size > 1:
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

        if self.in_features % self.world_size != 0:
            raise ValueError("in_features must be divisible by TP world size")
        self.in_features_per_shard = self.in_features // self.world_size
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features_per_shard))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)
        if init:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
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
            init=False,
        ).to(linear_module.weight.device, dtype=linear_module.weight.dtype)
        rank = row_linear.rank
        world_size = row_linear.world_size
        with torch.no_grad():
            dim_per_shard = linear_module.in_features // world_size
            start, end = rank * dim_per_shard, (rank + 1) * dim_per_shard
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
