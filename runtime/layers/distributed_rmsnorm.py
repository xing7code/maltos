from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class _ReplicatedAllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        out = x.contiguous().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_output.contiguous().clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_input, None


class DistributedRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        *,
        tp_group: dist.ProcessGroup | None = None,
        logical_hidden_size: int | None = None,
        init_weight: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.logical_hidden_size = logical_hidden_size or hidden_size
        self.eps = eps
        self.tp_group = tp_group
        self.weight = nn.Parameter(torch.ones(hidden_size) if init_weight else torch.empty(hidden_size))

    @property
    def world_size(self) -> int:
        if self.tp_group is None or not dist.is_initialized():
            return 1
        return dist.get_world_size(self.tp_group)

    @property
    def rank(self) -> int:
        if self.tp_group is None or not dist.is_initialized():
            return 0
        return dist.get_rank(self.tp_group)

    @classmethod
    def from_module(cls, module: nn.Module, tp_group: dist.ProcessGroup) -> "DistributedRMSNorm":
        if not hasattr(module, "weight") or not hasattr(module, "eps"):
            raise TypeError("DistributedRMSNorm.from_module expects an RMSNorm-like module with weight and eps")
        full_weight = module.weight.detach()
        world_size = dist.get_world_size(tp_group)
        if full_weight.numel() % world_size != 0:
            raise ValueError("RMSNorm hidden size must be divisible by TP world size")
        shard = full_weight.numel() // world_size
        rank = dist.get_rank(tp_group)
        out = cls(
            shard,
            float(module.eps),
            tp_group=tp_group,
            logical_hidden_size=full_weight.numel(),
            init_weight=False,
        ).to(device=full_weight.device, dtype=full_weight.dtype)
        with torch.no_grad():
            out.weight.copy_(full_weight[rank * shard : (rank + 1) * shard])
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_sq = x.float().pow(2).sum(dim=-1, keepdim=True)
        if self.world_size > 1:
            global_sq = _ReplicatedAllReduceSum.apply(local_sq, self.tp_group)
            variance = global_sq / self.logical_hidden_size
        else:
            variance = local_sq / self.hidden_size
        x_norm = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return self.weight * x_norm
