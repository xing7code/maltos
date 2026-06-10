from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn

from runtime.buffer_allocator import allocate_buffer

if TYPE_CHECKING:
    from runtime.core import StepContext


class _RingShift(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        group: dist.ProcessGroup,
        send_to: int,
        recv_from: int,
        alloc_key: str,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.send_to = send_to
        ctx.recv_from = recv_from
        ctx.alloc_key = alloc_key
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        if ctx.world_size == 1:
            return x
        out = allocate_buffer(
            key=f"{alloc_key}.forward",
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
        )
        send_global_rank = dist.get_global_rank(group, send_to)
        recv_global_rank = dist.get_global_rank(group, recv_from)
        _pairwise_send_recv(
            x.contiguous(),
            out,
            send_rank=send_global_rank,
            recv_rank=recv_global_rank,
            rank=ctx.rank,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.world_size == 1:
            return grad_output, None, None, None, None
        grad_input = allocate_buffer(
            key=f"{ctx.alloc_key}.backward",
            shape=tuple(grad_output.shape),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        send_global_rank = dist.get_global_rank(ctx.group, ctx.recv_from)
        recv_global_rank = dist.get_global_rank(ctx.group, ctx.send_to)
        _pairwise_send_recv(
            grad_output.contiguous(),
            grad_input,
            send_rank=send_global_rank,
            recv_rank=recv_global_rank,
            rank=ctx.rank,
        )
        return grad_input, None, None, None, None


def _ring_shift(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    send_to: int,
    recv_from: int,
    *,
    alloc_key: str,
) -> torch.Tensor:
    return _RingShift.apply(x, group, send_to, recv_from, alloc_key)


def _pairwise_send_recv(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    *,
    send_rank: int,
    recv_rank: int,
    rank: int,
) -> None:
    if rank % 2 == 0:
        dist.send(send_tensor, dst=send_rank)
        dist.recv(recv_tensor, src=recv_rank)
        return
    dist.recv(recv_tensor, src=recv_rank)
    dist.send(send_tensor, dst=send_rank)


class RingAttentionCore(nn.Module):
    def __init__(self, group: dist.ProcessGroup, step_context: "StepContext | None" = None) -> None:
        super().__init__()
        self.group = group
        self._step_context = step_context

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_offset: int,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        world_size = dist.get_world_size(self.group)
        if world_size == 1:
            return _ring_block_causal_attention(
                q,
                k,
                v,
                position_offset=position_offset,
                position_ids=position_ids,
            )

        rank = dist.get_rank(self.group)
        send_to = (rank + 1) % world_size
        recv_from = (rank - 1 + world_size) % world_size
        q_positions = _canonical_positions(
            position_ids,
            position_offset=position_offset,
            seq_len=q.size(-2),
            device=q.device,
        )
        current_kv = torch.cat([k, v], dim=-1)
        current_positions = q_positions
        running_max = torch.full(
            q.shape[:-1],
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
        running_lse = torch.zeros(
            q.shape[:-1],
            dtype=torch.float32,
            device=q.device,
        )
        running_acc = torch.zeros(
            (*q.shape[:-1], v.size(-1)),
            dtype=torch.float32,
            device=q.device,
        )

        mb_idx = self._step_context.pp_cur_microbatch_idx if self._step_context is not None else 0
        for step in range(world_size):
            current_k, current_v = current_kv.split(k.size(-1), dim=-1)
            running_max, running_lse, running_acc = _update_online_attention_state(
                q=q,
                k=current_k,
                v=current_v,
                q_positions=q_positions,
                key_positions=current_positions,
                running_max=running_max,
                running_lse=running_lse,
                running_acc=running_acc,
            )
            if step + 1 == world_size:
                break
            current_kv = _ring_shift(
                current_kv,
                self.group,
                send_to,
                recv_from,
                alloc_key=f"cp.ring.{id(self)}.mb{mb_idx}.step_{step}",
            )
            current_positions = _ring_exchange_tensor(
                current_positions,
                self.group,
                send_to,
                recv_from,
                alloc_key=f"cp.ring.{id(self)}.positions.step_{step}",
            )

        return (running_acc / running_lse.clamp_min(1e-20).unsqueeze(-1)).to(dtype=v.dtype)


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


def _ring_exchange_tensor(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    send_to: int,
    recv_from: int,
    *,
    alloc_key: str,
) -> torch.Tensor:
    out = allocate_buffer(
        key=alloc_key,
        shape=tuple(x.shape),
        dtype=x.dtype,
        device=x.device,
    )
    send_global_rank = dist.get_global_rank(group, send_to)
    recv_global_rank = dist.get_global_rank(group, recv_from)
    _pairwise_send_recv(
        x.contiguous(),
        out,
        send_rank=send_global_rank,
        recv_rank=recv_global_rank,
        rank=dist.get_rank(group),
    )
    return out


def _ring_block_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    position_offset: int,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    q_positions = _canonical_positions(
        position_ids,
        position_offset=position_offset,
        seq_len=q.size(-2),
        device=q.device,
    )
    running_max = torch.full(
        q.shape[:-1],
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    running_lse = torch.zeros(
        q.shape[:-1],
        dtype=torch.float32,
        device=q.device,
    )
    running_acc = torch.zeros(
        (*q.shape[:-1], v.size(-1)),
        dtype=torch.float32,
        device=q.device,
    )
    running_max, running_lse, running_acc = _update_online_attention_state(
        q=q,
        k=k,
        v=v,
        q_positions=q_positions,
        key_positions=q_positions,
        running_max=running_max,
        running_lse=running_lse,
        running_acc=running_acc,
    )
    return (running_acc / running_lse.clamp_min(1e-20).unsqueeze(-1)).to(dtype=v.dtype)


def _update_online_attention_state(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_positions: torch.Tensor,
    key_positions: torch.Tensor,
    running_max: torch.Tensor,
    running_lse: torch.Tensor,
    running_acc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    causal_mask = key_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q.float(), k.transpose(-2, -1).float()) * scale
    scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    block_max = scores.max(dim=-1).values
    new_max = torch.maximum(running_max, block_max)

    prev_scale = torch.where(
        torch.isfinite(running_max),
        torch.exp(running_max - new_max),
        torch.zeros_like(new_max),
    )
    block_probs = torch.exp(scores - new_max.unsqueeze(-1))
    block_probs = torch.where(
        causal_mask.unsqueeze(0).unsqueeze(0),
        block_probs,
        torch.zeros_like(block_probs),
    )
    new_lse = running_lse * prev_scale + block_probs.sum(dim=-1)
    new_acc = running_acc * prev_scale.unsqueeze(-1) + torch.matmul(block_probs, v.float())
    return new_max, new_lse, new_acc
