from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from absl import logging

from runtime.layers.attn_masking_utils import build_example_causal_mask, canonical_position_ids, canonical_sequence_ids
from runtime.layers.flash_utils import flash_attn_dense_block_fallback_reason
from runtime.layers.functional import flash_ring_attention, ring_shift
from runtime.layers.ring_layout import has_zigzag_ring_layout
from utils.attention_backend import AttentionBackend, validate_attention_backend

if TYPE_CHECKING:
    from runtime.types import StepContext


class RingAttentionCore(nn.Module):
    _warned_flash_attn_fallback_global = False
    _warned_non_flash_backend_fallback_global = False

    def __init__(
        self,
        group: dist.ProcessGroup,
        step_context: "StepContext | None" = None,
        attention_backend: str = AttentionBackend.EAGER,
    ) -> None:
        super().__init__()
        self.group = group
        self._step_context = step_context
        self.attention_backend = validate_attention_backend(attention_backend)

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
            return _single_rank_ring_attention(
                q,
                k,
                v,
                position_offset=position_offset,
                position_ids=position_ids,
                sequence_ids=sequence_ids,
            )

        q_positions = canonical_position_ids(
            position_ids,
            batch_size=q.size(0),
            seq_len=q.size(-2),
            position_offset=position_offset,
            device=q.device,
        )
        q_sequence_ids = canonical_sequence_ids(
            sequence_ids,
            batch_size=q.size(0),
            seq_len=q.size(-2),
            device=q.device,
        )
        rank = dist.get_rank(self.group)
        flash_ring_reason = _flash_ring_fallback_reason(
            q=q,
            q_positions=q_positions,
            q_sequence_ids=q_sequence_ids,
            rank=rank,
            world_size=world_size,
        )
        if self.attention_backend == AttentionBackend.FLASH_ATTN and flash_ring_reason is None:
            mb_idx = self._step_context.pp_cur_microbatch_idx if self._step_context is not None else 0
            return flash_ring_attention(
                q,
                k,
                v,
                self.group,
                module_id=id(self),
                mb_idx=mb_idx,
            )
        if self.attention_backend == AttentionBackend.FLASH_ATTN:
            if not type(self)._warned_flash_attn_fallback_global:
                logging.warning(
                    "RingAttentionCore flash_attn backend falling back to eager ring attention: %s",
                    flash_ring_reason,
                )
                type(self)._warned_flash_attn_fallback_global = True
        elif self.attention_backend != AttentionBackend.EAGER:
            if not type(self)._warned_non_flash_backend_fallback_global:
                logging.warning(
                    "RingAttentionCore backend %s is unsupported; using eager ring attention",
                    self.attention_backend,
                )
                type(self)._warned_non_flash_backend_fallback_global = True
        return _eager_ring_attention(
            q,
            k,
            v,
            q_positions=q_positions,
            q_sequence_ids=q_sequence_ids,
            group=self.group,
            module_id=id(self),
            mb_idx=self._step_context.pp_cur_microbatch_idx if self._step_context is not None else 0,
        )


def _single_rank_ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    position_offset: int,
    position_ids: torch.Tensor | None = None,
    sequence_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    q_positions = canonical_position_ids(
        position_ids,
        batch_size=q.size(0),
        seq_len=q.size(-2),
        position_offset=position_offset,
        device=q.device,
    )
    q_sequence_ids = canonical_sequence_ids(
        sequence_ids,
        batch_size=q.size(0),
        seq_len=q.size(-2),
        device=q.device,
    )
    running_max = torch.full(q.shape[:-1], float("-inf"), dtype=torch.float32, device=q.device)
    running_lse = torch.zeros(q.shape[:-1], dtype=torch.float32, device=q.device)
    running_acc = torch.zeros((*q.shape[:-1], v.size(-1)), dtype=torch.float32, device=q.device)
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


def _eager_ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_positions: torch.Tensor,
    q_sequence_ids: torch.Tensor | None,
    group: dist.ProcessGroup,
    module_id: int,
    mb_idx: int,
) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    send_to = (rank + 1) % world_size
    recv_from = (rank - 1 + world_size) % world_size
    current_kv = torch.cat([k, v], dim=-1)
    current_positions = q_positions
    current_sequence_ids = q_sequence_ids
    running_max = torch.full(q.shape[:-1], float("-inf"), dtype=torch.float32, device=q.device)
    running_lse = torch.zeros(q.shape[:-1], dtype=torch.float32, device=q.device)
    running_acc = torch.zeros((*q.shape[:-1], v.size(-1)), dtype=torch.float32, device=q.device)

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
            group,
            send_to,
            recv_from,
            alloc_key=f"cp.ring.{module_id}.mb{mb_idx}.step_{step}",
        )
        current_positions = _ring_exchange_tensor(
            current_positions,
            group,
            send_to,
            recv_from,
            alloc_key=f"cp.ring.{module_id}.positions.step_{step}",
        )
        if current_sequence_ids is not None:
            current_sequence_ids = _ring_exchange_tensor(
                current_sequence_ids,
                group,
                send_to,
                recv_from,
                alloc_key=f"cp.ring.{module_id}.sequence_ids.step_{step}",
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


def _ring_exchange_tensor(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    send_to: int,
    recv_from: int,
    *,
    alloc_key: str,
) -> torch.Tensor:
    return ring_shift(x, group, send_to, recv_from, alloc_key=alloc_key)


def _flash_ring_fallback_reason(
    *,
    q: torch.Tensor,
    q_positions: torch.Tensor,
    q_sequence_ids: torch.Tensor | None,
    rank: int,
    world_size: int,
) -> str | None:
    reason = flash_attn_dense_block_fallback_reason(q)
    if reason is not None:
        return reason
    if q_sequence_ids is not None:
        return "flash ring fast path currently supports only dense sequences; packed sequence_ids use eager ring attention"
    if not has_zigzag_ring_layout(
        q_positions,
        rank=rank,
        world_size=world_size,
    ):
        return "flash ring fast path requires canonical zigzag local positions"
    return None
