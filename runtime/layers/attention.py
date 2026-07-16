from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from runtime.layers.functional import all_gather, ring_shift
from utils.attention_masking import build_example_causal_mask, canonical_position_ids, canonical_sequence_ids

if TYPE_CHECKING:
    from runtime.types import StepContext


class AllGatherKvAttentionCore(nn.Module):
    def __init__(self, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.group = group

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_offset: int,
        position_ids: torch.Tensor | None = None,
        sequence_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gathered_k = all_gather(
            k,
            self.group,
            comm_dim=2,
            alloc_key=f"cp.all_gather_kv.{id(self)}.k",
            backward_reduce_op=dist.ReduceOp.SUM,
        )
        gathered_v = all_gather(
            v,
            self.group,
            comm_dim=2,
            alloc_key=f"cp.all_gather_kv.{id(self)}.v",
            backward_reduce_op=dist.ReduceOp.SUM,
        )
        local_positions = _canonical_positions(
            position_ids,
            position_offset=position_offset,
            batch_size=q.size(0),
            seq_len=q.size(-2),
            device=q.device,
        )
        local_sequence_ids = _canonical_sequence_ids(
            sequence_ids,
            batch_size=q.size(0),
            seq_len=q.size(-2),
            device=q.device,
        )
        gathered_positions = [torch.empty_like(local_positions) for _ in range(dist.get_world_size(self.group))]
        dist.all_gather(gathered_positions, local_positions.contiguous(), group=self.group)
        gathered_sequence_ids = None
        if local_sequence_ids is not None:
            gathered_sequence_ids = [torch.empty_like(local_sequence_ids) for _ in range(dist.get_world_size(self.group))]
            dist.all_gather(gathered_sequence_ids, local_sequence_ids.contiguous(), group=self.group)
        return _eager_cp_causal_attention(
            q,
            gathered_k,
            gathered_v,
            q_positions=local_positions,
            k_positions=torch.cat(gathered_positions, dim=1),
            q_sequence_ids=local_sequence_ids,
            k_sequence_ids=None if gathered_sequence_ids is None else torch.cat(gathered_sequence_ids, dim=1),
        )


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
        sequence_ids: torch.Tensor | None = None,
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
            batch_size=q.size(0),
            seq_len=q.size(-2),
            device=q.device,
        )
        q_sequence_ids = _canonical_sequence_ids(
            sequence_ids,
            batch_size=q.size(0),
            seq_len=q.size(-2),
            device=q.device,
        )
        current_kv = torch.cat([k, v], dim=-1)
        current_positions = q_positions
        current_sequence_ids = q_sequence_ids
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
                q_sequence_ids=q_sequence_ids,
                key_sequence_ids=current_sequence_ids,
                running_max=running_max,
                running_lse=running_lse,
                running_acc=running_acc,
            )
            if step + 1 == world_size:
                break
            current_kv = ring_shift(
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
            if current_sequence_ids is not None:
                current_sequence_ids = _ring_exchange_tensor(
                    current_sequence_ids,
                    self.group,
                    send_to,
                    recv_from,
                    alloc_key=f"cp.ring.{id(self)}.sequence_ids.step_{step}",
                )

        return (running_acc / running_lse.clamp_min(1e-20).unsqueeze(-1)).to(dtype=v.dtype)


def _canonical_positions(
    position_ids: torch.Tensor | None,
    *,
    position_offset: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    return canonical_position_ids(
        position_ids,
        batch_size=batch_size,
        seq_len=seq_len,
        position_offset=position_offset,
        device=device,
    )


def _canonical_sequence_ids(
    sequence_ids: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    return canonical_sequence_ids(
        sequence_ids,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )


def _eager_cp_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    q_sequence_ids: torch.Tensor | None = None,
    k_sequence_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    causal_mask = build_example_causal_mask(
        q_positions=q_positions,
        k_positions=k_positions,
        q_sequence_ids=q_sequence_ids,
        k_sequence_ids=k_sequence_ids,
    )
    scores = scores.masked_fill(~causal_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def _ring_exchange_tensor(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    send_to: int,
    recv_from: int,
    *,
    alloc_key: str,
) -> torch.Tensor:
    return ring_shift(x, group, send_to, recv_from, alloc_key=alloc_key)


def _ring_block_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    position_offset: int,
    position_ids: torch.Tensor | None = None,
    sequence_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    q_positions = _canonical_positions(
        position_ids,
        position_offset=position_offset,
        batch_size=q.size(0),
        seq_len=q.size(-2),
        device=q.device,
    )
    q_sequence_ids = _canonical_sequence_ids(
        sequence_ids,
        batch_size=q.size(0),
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
        q_sequence_ids=q_sequence_ids,
        key_sequence_ids=q_sequence_ids,
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
    q_sequence_ids: torch.Tensor | None,
    key_sequence_ids: torch.Tensor | None,
    running_max: torch.Tensor,
    running_lse: torch.Tensor,
    running_acc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    causal_mask = build_example_causal_mask(
        q_positions=q_positions,
        k_positions=key_positions,
        q_sequence_ids=q_sequence_ids,
        k_sequence_ids=key_sequence_ids,
    )
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q.float(), k.transpose(-2, -1).float()) * scale
    scores = scores.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))
    block_max = scores.max(dim=-1).values
    new_max = torch.maximum(running_max, block_max)

    prev_scale = torch.where(
        torch.isfinite(running_max),
        torch.exp(running_max - new_max),
        torch.zeros_like(new_max),
    )
    block_probs = torch.exp(scores - new_max.unsqueeze(-1))
    block_probs = torch.where(
        causal_mask.unsqueeze(1),
        block_probs,
        torch.zeros_like(block_probs),
    )
    new_lse = running_lse * prev_scale + block_probs.sum(dim=-1)
    new_acc = running_acc * prev_scale.unsqueeze(-1) + torch.matmul(block_probs, v.float())
    return new_max, new_lse, new_acc
