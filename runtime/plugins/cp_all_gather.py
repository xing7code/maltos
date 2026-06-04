from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class _ContextParallelMeta:
    group: dist.ProcessGroup


class AllGatherKvAttentionCore(nn.Module):
    def __init__(self, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.meta = _ContextParallelMeta(group)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, position_offset: int) -> torch.Tensor:
        gathered_k = _cp_all_gather(k, self.meta.group, comm_dim=2)
        gathered_v = _cp_all_gather(v, self.meta.group, comm_dim=2)
        return _eager_cp_causal_attention(q, gathered_k, gathered_v, position_offset=position_offset)


def _eager_cp_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    position_offset: int,
) -> torch.Tensor:
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    local_seq = q.size(-2)
    global_seq = k.size(-2)
    q_positions = torch.arange(position_offset, position_offset + local_seq, device=q.device)
    k_positions = torch.arange(global_seq, device=q.device)
    causal_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
    scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


class _ContextParallelAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup, comm_dim: int) -> torch.Tensor:
        ctx.group = group
        ctx.comm_dim = comm_dim
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        out = [torch.empty_like(x) for _ in range(ctx.world_size)]
        dist.all_gather(out, x.contiguous(), group=group)
        return torch.cat(out, dim=comm_dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        per_rank_dim = grad_output.shape[ctx.comm_dim] // ctx.world_size
        grad = grad_output.narrow(ctx.comm_dim, ctx.rank * per_rank_dim, per_rank_dim).contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad, None, None


def _cp_all_gather(x: torch.Tensor, group: dist.ProcessGroup, comm_dim: int) -> torch.Tensor:
    return _ContextParallelAllGather.apply(x, group, comm_dim)
