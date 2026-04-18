import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from enum import Enum

from train_system.runtime.plugin import BaseParallelPlugin


_COMM_BUFFER_CAP_RATIO = 1.5

class ColumnParallelLinear(nn.Module):

    def __init__(self, in_features, out_features, tp_group, bias=True, gather_output=True, init=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        self.tp_group = tp_group
        self.rank = dist.get_rank(tp_group)
        self.world_size = dist.get_world_size(tp_group)

        self.register_buffer("_gather_buf_1d", torch.empty(0), persistent=False)
        self._buf_meta = (None, None) # dtype, device

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
            gathered_numel = output.numel() * self.world_size
            if output.dtype != self._buf_meta[0] or output.device != self._buf_meta[1] or gathered_numel > self._gather_buf_1d.numel():
                new_cap = max(gathered_numel, int(self._gather_buf_1d.numel() * _COMM_BUFFER_CAP_RATIO))
                self._gather_buf_1d = torch.empty(new_cap, dtype=output.dtype, device=output.device)
                self._buf_meta = (output.dtype, output.device)
            shard_shape = output.size()
            dist.all_gather_into_tensor(self._gather_buf_1d[:gathered_numel], output.reshape(-1), self.tp_group)
            output = torch.cat([x.view(shard_shape) for x in self._gather_buf_1d[:gathered_numel].chunk(self.world_size)], dim=-1)
        return output


class RowParallelCollective(str, Enum):
    REDUCE = "all_reduce"
    SCATTER_DIM1 = "reduce_scatter_dim1"
    NONE = "none"


class RowParallelLinear(nn.Module):

    def __init__(self, in_features, out_features, tp_group, collective=RowParallelCollective.NONE, bias=True, init=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.tp_group = tp_group
        self.collective = collective
        self.rank = dist.get_rank(tp_group)
        self.world_size = dist.get_world_size(tp_group)

        self.register_buffer("_scatter_buf_1d", torch.empty(0), persistent=False)
        self._scatter_meta = (None, None)

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
    def from_linear(cls, linear_module, tp_group, collective):
        row_linear = cls(
            linear_module.in_features,
            linear_module.out_features,
            tp_group,
            collective,
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
        if self.collective == RowParallelCollective.REDUCE:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)
        elif self.collective == RowParallelCollective.SCATTER_DIM1:
            scatter_numel = output.numel() // self.world_size
            scatter_dim = output.size(1)
            assert scatter_dim % self.world_size == 0, f"For SCATTER_DIM1 mode in RowParallelLinear, output dim1 {scatter_dim} must be divisible by world_size {self.world_size}"
            orig_shape = output.size()
            if output.dtype != self._scatter_meta[0] or output.device != self._scatter_meta[1] or scatter_numel > self._scatter_buf_1d.numel():
                new_cap = max(scatter_numel, int(self._scatter_buf_1d.numel() * _COMM_BUFFER_CAP_RATIO))
                self._scatter_buf_1d = torch.empty(new_cap, dtype=output.dtype, device=output.device)
                self._scatter_meta = (output.dtype, output.device)
            output = output.transpose(0,1).reshape(-1)
            dist.reduce_scatter_tensor(self._scatter_buf_1d[:scatter_numel].detach(), output, op=dist.ReduceOp.SUM, group=self.tp_group)
            output = self._scatter_buf_1d[:scatter_numel].reshape(scatter_dim//self.world_size, orig_shape[0], *orig_shape[2:]).transpose(0,1)
        if self.collective in (RowParallelCollective.REDUCE, RowParallelCollective.SCATTER_DIM1) and self.bias is not None:
            output = output + self.bias
        return output


class TpPlugin(BaseParallelPlugin):

    _ROW_COLLECTIVE_MAP = {
        "all_reduce": RowParallelCollective.REDUCE,
        "reduce_scatter": RowParallelCollective.SCATTER_DIM1,
        "none": RowParallelCollective.NONE,
    }
    _COL_POST_COMM = {"all_gather", "none"}
    _ROW_POST_COMM = set(_ROW_COLLECTIVE_MAP.keys())
    _SUPPORTED_PRE_COMM = {"none"}

    def __init__(self, tp_group: dist.ProcessGroup):
        self.tp_group = tp_group

    def setup_model(self, model: nn.Module) -> nn.Module:
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
                    RowParallelLinear.from_linear(mod, self.tp_group, self._ROW_COLLECTIVE_MAP[r.post_comm])
                )
        return model
