from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from runtime.buffer_allocator import allocate_buffer
from runtime.layers.functional import all_gather, all_reduce, reduce_scatter


class _RowParallelReduceScatterAsync(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, tp_group, alloc_key: str):
        ctx.tp_group = tp_group
        ctx.alloc_key = alloc_key
        ctx.use_bias = bias is not None
        ctx.input_shape = tuple(input.shape)
        ctx.rank = dist.get_rank(tp_group)
        ctx.world_size = dist.get_world_size(tp_group)
        ctx.save_for_backward(input, weight)

        output = F.linear(input, weight, None)
        output = reduce_scatter(
            output,
            tp_group,
            1,
            alloc_key=f"{alloc_key}.forward.reduce_scatter",
        )
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_output_weight = weight.to(dtype=grad_output.dtype)
        grad_weight_dtype = weight.dtype

        grad_output_t = grad_output.transpose(0, 1).contiguous()
        gathered_shape = (ctx.world_size * grad_output_t.shape[0], *grad_output_t.shape[1:])
        gathered_grad_output_t = allocate_buffer(
            key=f"{ctx.alloc_key}.backward.all_gather",
            shape=gathered_shape,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        handle = dist.all_gather_into_tensor(
            gathered_grad_output_t,
            grad_output_t,
            group=ctx.tp_group,
            async_op=True,
        )

        grad_input = torch.empty(ctx.input_shape, dtype=grad_output.dtype, device=grad_output.device)
        local_seq = grad_output.shape[1]
        local_start = ctx.rank * local_seq
        local_end = local_start + local_seq
        grad_input.narrow(1, local_start, local_seq).copy_(grad_output.matmul(grad_output_weight))

        handle.wait()

        gathered_grad_output = gathered_grad_output_t.transpose(0, 1)
        if local_start > 0:
            left = gathered_grad_output.narrow(1, 0, local_start)
            grad_input.narrow(1, 0, local_start).copy_(left.matmul(grad_output_weight))
        if local_end < grad_input.shape[1]:
            right = gathered_grad_output.narrow(1, local_end, grad_input.shape[1] - local_end)
            grad_input.narrow(1, local_end, grad_input.shape[1] - local_end).copy_(right.matmul(grad_output_weight))

        grad_weight = gathered_grad_output.reshape(-1, gathered_grad_output.shape[-1]).to(grad_weight_dtype).t().matmul(
            input.reshape(-1, input.shape[-1]).to(grad_weight_dtype)
        )
        grad_bias = gathered_grad_output.to(grad_weight_dtype).sum(dim=(0, 1)) if ctx.use_bias else None
        return grad_input, grad_weight, grad_bias, None, None


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
            output = all_gather(
                output,
                self.tp_group,
                -1,
                alloc_key=f"tp.column_parallel_linear.{id(self)}.all_gather",
            )
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
        if self.comm == "reduce_scatter":
            if self.world_size > 1:
                return _RowParallelReduceScatterAsync.apply(
                    input,
                    self.weight,
                    self.bias,
                    self.tp_group,
                    f"tp.row_parallel_linear.{id(self)}",
                )
            output = F.linear(input, self.weight, None)
            if self.bias is not None:
                output = output + self.bias
            return output

        output = F.linear(input, self.weight, None)
        if self.comm == "all_reduce":
            output = all_reduce(output, self.tp_group, dist.ReduceOp.SUM)
        if self.bias is not None:
            output = output + self.bias
        return output
