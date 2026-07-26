from __future__ import annotations

"""Dynamic subgroup Ring Attention for the HDP-balanced baseline.

Every rank shares one DCP process group.  A document names a subset of its
members in the ByteScale schedule; only those ranks exchange its
KV blocks.  In particular, this intentionally does *not* call
``dist.new_group`` per document, which would make a variable-length batch
both expensive and fragile.
"""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn

from runtime.layers.attn_masking_utils import canonical_position_ids, canonical_sequence_ids
from runtime.layers.cp_functional import dynamic_flash_ring_attention, subset_ring_exchange_metadata, subset_ring_shift
from runtime.layers.flash_utils import flash_attn_block_fallback_reason
from runtime.layers.ring_attention import _update_online_attention_state
from utils.attention_backend import AttentionBackend, validate_attention_backend
from utils.profiling import profiled

if TYPE_CHECKING:
    from parallel.hdp_helper import ByteScaleHdpLocalLayout


class HdpBalancedAttentionCore(nn.Module):
    """Eager, autograd-correct dynamic-ring attention.

    This is intentionally a correctness core, not FCP's optimized block
    pipeline.  It provides the exact recursive-ring semantics required for
    the HDP-balanced baseline and is the reference against which the future
    Flash/P2P overlap implementation will be checked.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        attention_backend: str = AttentionBackend.EAGER,
    ) -> None:
        super().__init__()
        self.group = group
        self._active_layout: ByteScaleHdpLocalLayout | None = None
        self.preserves_native_gqa = True
        self.attention_backend = validate_attention_backend(attention_backend)

    def set_active_schedule(self, layout: "ByteScaleHdpLocalLayout") -> None:
        self._active_layout = layout

    @profiled("maltos::hdp_balanced.attention")
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_offset: int,
        position_ids: torch.Tensor | None = None,
        sequence_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layout = self._active_layout
        if layout is None:
            raise RuntimeError("HdpBalancedAttentionCore requires an active ByteScaleHdpLocalLayout")
        rank = dist.get_rank(self.group)
        if rank != layout.rank:
            raise RuntimeError(f"HDP attention layout rank={layout.rank} does not match process-group rank={rank}")
        if not layout.documents:
            return q * 0 + (k.sum() + v.sum()) * 0
        if q.size(0) != layout.packed_rows:
            raise ValueError("HDP attention batch rows must equal local packed rows")
        _validate_hdp_qkv(q, k, v)
        if self.attention_backend not in (AttentionBackend.EAGER, AttentionBackend.FLASH_ATTN):
            raise ValueError(
                "HDP dynamic-subset RingAttention supports only eager or flash_attn, got "
                f"{self.attention_backend!r}"
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
        output = torch.zeros_like(q)
        # Every participant independently gets this same planner ordering.
        for document in layout.documents:
            row_slice = slice(document.local_row, document.local_row + 1)
            token_slice = slice(document.local_start, document.local_end)
            row_q = q[row_slice, :, token_slice, :]
            row_k = k[row_slice, :, token_slice, :]
            row_v = v[row_slice, :, token_slice, :]
            row_positions = q_positions[row_slice, token_slice]
            row_sequence_ids = None if q_sequence_ids is None else q_sequence_ids[row_slice, token_slice]
            if self.attention_backend == AttentionBackend.FLASH_ATTN:
                reason = flash_attn_block_fallback_reason(row_q)
                if reason is not None:
                    raise RuntimeError(
                        "HDP flash_attn backend is unavailable for dynamic-subset Flash Ring: " + reason
                    )
                row_out = dynamic_flash_ring_attention(
                    row_q, row_k, row_v,
                    positions=row_positions,
                    sequence_ids=row_sequence_ids,
                    group=self.group,
                    participant_ranks=document.participant_ranks,
                    module_id=id(self),
                )
            elif len(document.participant_ranks) == 1:
                row_out = _single_document_attention(
                    row_q,
                    row_k,
                    row_v,
                    positions=row_positions,
                    sequence_ids=row_sequence_ids,
                )
            else:
                row_out = _dynamic_eager_ring_attention(
                    row_q,
                    row_k,
                    row_v,
                    positions=row_positions,
                    sequence_ids=row_sequence_ids,
                    group=self.group,
                    participant_ranks=document.participant_ranks,
                )
            output[row_slice, :, token_slice, :] = row_out
        return output


def _single_document_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    positions: torch.Tensor,
    sequence_ids: torch.Tensor | None,
) -> torch.Tensor:
    k, v = _expand_kv_for_query_heads(q, k, v)
    running_max = torch.full(q.shape[:-1], float("-inf"), dtype=torch.float32, device=q.device)
    running_lse = torch.zeros(q.shape[:-1], dtype=torch.float32, device=q.device)
    running_acc = torch.zeros((*q.shape[:-1], v.size(-1)), dtype=torch.float32, device=q.device)
    _, running_lse, running_acc = _update_online_attention_state(
        q=q,
        k=k,
        v=v,
        q_positions=positions,
        key_positions=positions,
        q_sequence_ids=sequence_ids,
        key_sequence_ids=sequence_ids,
        running_max=running_max,
        running_lse=running_lse,
        running_acc=running_acc,
    )
    return (running_acc / running_lse.clamp_min(1e-20).unsqueeze(-1)).to(dtype=v.dtype)


def _validate_hdp_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("HDP attention expects q/k/v shaped [batch, heads, tokens, head_dim]")
    if k.shape != v.shape:
        raise ValueError(f"HDP attention K/V shapes must match, got {tuple(k.shape)} and {tuple(v.shape)}")
    if q.size(0) != k.size(0) or q.size(2) != k.size(2) or q.size(3) != k.size(3):
        raise ValueError("HDP attention Q/K/V batch, token, and head-dim sizes must match")
    if k.size(1) < 1 or q.size(1) < 1 or q.size(1) % k.size(1):
        raise ValueError("HDP attention Q heads must be a positive multiple of native KV heads")


def _dynamic_eager_ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    positions: torch.Tensor,
    sequence_ids: torch.Tensor | None,
    group: dist.ProcessGroup,
    participant_ranks: tuple[int, ...],
) -> torch.Tensor:
    current_kv = torch.cat((k, v), dim=-1)
    current_positions = positions
    current_sequence_ids = sequence_ids
    running_max = torch.full(q.shape[:-1], float("-inf"), dtype=torch.float32, device=q.device)
    running_lse = torch.zeros(q.shape[:-1], dtype=torch.float32, device=q.device)
    running_acc = torch.zeros((*q.shape[:-1], v.size(-1)), dtype=torch.float32, device=q.device)

    for step in range(len(participant_ranks)):
        current_k, current_v = current_kv.split(k.size(-1), dim=-1)
        compute_k, compute_v = _expand_kv_for_query_heads(q, current_k, current_v)
        running_max, running_lse, running_acc = _update_online_attention_state(
            q=q,
            k=compute_k,
            v=compute_v,
            q_positions=positions,
            key_positions=current_positions,
            q_sequence_ids=sequence_ids,
            key_sequence_ids=current_sequence_ids,
            running_max=running_max,
            running_lse=running_lse,
            running_acc=running_acc,
        )
        if step + 1 == len(participant_ranks):
            break
        current_kv = subset_ring_shift(current_kv, group=group, participant_ranks=participant_ranks)
        current_positions = subset_ring_exchange_metadata(
            current_positions,
            group=group,
            participant_ranks=participant_ranks,
        )
        if current_sequence_ids is not None:
            current_sequence_ids = subset_ring_exchange_metadata(
                current_sequence_ids,
                group=group,
                participant_ranks=participant_ranks,
            )
    return (running_acc / running_lse.clamp_min(1e-20).unsqueeze(-1)).to(dtype=v.dtype)


def _expand_kv_for_query_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand native GQA K/V only for local eager attention computation."""
    if q.size(1) == k.size(1):
        return k, v
    if q.size(1) % k.size(1):
        raise ValueError("HDP attention Q heads must be divisible by native KV heads")
    repeats = q.size(1) // k.size(1)
    return k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
