from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from runtime.layers.functional import all_gather


@dataclass
class _ContextParallelMeta:
    group: dist.ProcessGroup


class AllGatherKvAttentionCore(nn.Module):
    def __init__(self, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.meta = _ContextParallelMeta(group)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_offset: int,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gathered_k = all_gather(
            k,
            self.meta.group,
            comm_dim=2,
            alloc_key=f"cp.all_gather_kv.{id(self)}.k",
            backward_reduce_op=dist.ReduceOp.SUM,
        )
        gathered_v = all_gather(
            v,
            self.meta.group,
            comm_dim=2,
            alloc_key=f"cp.all_gather_kv.{id(self)}.v",
            backward_reduce_op=dist.ReduceOp.SUM,
        )
        local_positions = _canonical_positions(
            position_ids,
            position_offset=position_offset,
            seq_len=q.size(-2),
            device=q.device,
        )
        gathered_positions = [torch.empty_like(local_positions) for _ in range(dist.get_world_size(self.meta.group))]
        dist.all_gather(gathered_positions, local_positions.contiguous(), group=self.meta.group)
        return _eager_cp_causal_attention(
            q,
            gathered_k,
            gathered_v,
            q_positions=local_positions,
            k_positions=torch.cat(gathered_positions, dim=0),
        )


def _canonical_positions(
    position_ids: torch.Tensor | None,
    *,
    position_offset: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if position_ids is None:
        return torch.arange(position_offset, position_offset + seq_len, device=device, dtype=torch.long)
    if position_ids.dim() == 1:
        positions = position_ids
    elif position_ids.dim() == 2:
        positions = position_ids[0]
        if position_ids.size(0) > 1 and not torch.equal(position_ids, positions.unsqueeze(0).expand_as(position_ids)):
            raise ValueError("ContextParallel attention expects identical position_ids across batch dimension")
    else:
        raise ValueError(f"position_ids must have rank 1 or 2, got shape={tuple(position_ids.shape)}")
    if positions.numel() != seq_len:
        raise ValueError(f"position_ids length must match local sequence length, got {positions.numel()} vs {seq_len}")
    return positions.to(device=device, dtype=torch.long)


def _eager_cp_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
) -> torch.Tensor:
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    causal_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
    scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
