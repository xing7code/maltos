"""Autograd-aware point-to-point primitives for context parallelism."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from runtime.buffer_allocator import BufferPolicy, acquire_buffer
from runtime.layers.flash_utils import (
    VarlenPrefixMetadata,
    flash_attn_varlen_prefix_backward,
    flash_attn_varlen_prefix_with_lse,
)
from utils.distributed import pairwise_send_recv_async


class _SubsetRingShift(torch.autograd.Function):
    """Shift a tensor around an arbitrary subset of an existing process group."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        group: dist.ProcessGroup,
        participant_ranks: tuple[int, ...],
    ) -> torch.Tensor:
        ctx.group = group
        ctx.participant_ranks = participant_ranks
        return _subset_ring_exchange(x, group=group, participant_ranks=participant_ranks)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        ranks: tuple[int, ...] = ctx.participant_ranks
        local_index = ranks.index(dist.get_rank(ctx.group))
        grad_input = torch.empty_like(grad_output)
        _pairwise_exchange(
            grad_output.contiguous(),
            grad_input,
            group=ctx.group,
            send_to=ranks[(local_index - 1) % len(ranks)],
            recv_from=ranks[(local_index + 1) % len(ranks)],
        )
        return grad_input, None, None


def subset_ring_shift(
    x: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    participant_ranks: tuple[int, ...],
) -> torch.Tensor:
    """Autograd-aware KV shift around a document-specific CP sub-ring."""
    return _SubsetRingShift.apply(x, group, participant_ranks)


def subset_ring_exchange_metadata(
    x: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    participant_ranks: tuple[int, ...],
) -> torch.Tensor:
    """Exchange non-differentiable position or sequence-id metadata."""
    return _subset_ring_exchange(x, group=group, participant_ranks=participant_ranks)


def _subset_ring_exchange(
    x: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    participant_ranks: tuple[int, ...],
) -> torch.Tensor:
    if len(participant_ranks) == 1:
        return x
    received = torch.empty_like(x)
    local_index = participant_ranks.index(dist.get_rank(group))
    _pairwise_exchange(
        x.contiguous(),
        received,
        group=group,
        send_to=participant_ranks[(local_index + 1) % len(participant_ranks)],
        recv_from=participant_ranks[(local_index - 1) % len(participant_ranks)],
    )
    return received


def _pairwise_exchange(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    send_to: int,
    recv_from: int,
) -> None:
    send_global_rank = dist.get_global_rank(group, send_to)
    recv_global_rank = dist.get_global_rank(group, recv_from)
    for work in pairwise_send_recv_async(
        send_tensor,
        recv_tensor,
        send_rank=send_global_rank,
        recv_rank=recv_global_rank,
        group=group,
    ):
        work.wait()


@dataclass
class _AsyncSubsetExchange:
    """A single tensor's async parent-group subset-ring exchange."""

    send_tensor: torch.Tensor
    recv_tensor: torch.Tensor
    works: list[object]

    def wait(self) -> torch.Tensor:
        with torch.profiler.record_function("maltos::hdp.flash.p2p.wait"):
            for work in self.works:
                work.wait()
        return self.recv_tensor


@dataclass
class _AsyncSubsetBlockExchange:
    """Double-buffered K/V plus document metadata exchange."""

    k: _AsyncSubsetExchange
    v: _AsyncSubsetExchange
    positions: _AsyncSubsetExchange
    sequence_ids: _AsyncSubsetExchange | None

    def wait(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return (
            self.k.wait(),
            self.v.wait(),
            self.positions.wait(),
            None if self.sequence_ids is None else self.sequence_ids.wait(),
        )


def _async_subset_ring_exchange(
    x: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    participant_ranks: tuple[int, ...],
    alloc_key: str,
) -> _AsyncSubsetExchange:
    if len(participant_ranks) == 1:
        return _AsyncSubsetExchange(x, x, [])
    local_index = participant_ranks.index(dist.get_rank(group))
    send_to = participant_ranks[(local_index + 1) % len(participant_ranks)]
    recv_from = participant_ranks[(local_index - 1) % len(participant_ranks)]
    send_tensor = x.contiguous()
    recv_tensor = acquire_buffer(
        shape=tuple(x.shape), dtype=x.dtype, device=x.device,
        policy=BufferPolicy.PINNED, key=alloc_key,
    ).tensor
    with torch.profiler.record_function("maltos::hdp.flash.p2p.launch"):
        works = pairwise_send_recv_async(
            send_tensor, recv_tensor,
            send_rank=dist.get_global_rank(group, send_to),
            recv_rank=dist.get_global_rank(group, recv_from),
            group=group,
        )
    return _AsyncSubsetExchange(send_tensor, recv_tensor, works)


def _async_subset_block_exchange(
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    sequence_ids: torch.Tensor | None,
    *,
    group: dist.ProcessGroup,
    participant_ranks: tuple[int, ...],
    alloc_key: str,
) -> _AsyncSubsetBlockExchange:
    return _AsyncSubsetBlockExchange(
        k=_async_subset_ring_exchange(k, group=group, participant_ranks=participant_ranks, alloc_key=f"{alloc_key}.k"),
        v=_async_subset_ring_exchange(v, group=group, participant_ranks=participant_ranks, alloc_key=f"{alloc_key}.v"),
        positions=_async_subset_ring_exchange(positions, group=group, participant_ranks=participant_ranks, alloc_key=f"{alloc_key}.positions"),
        sequence_ids=(
            None if sequence_ids is None else _async_subset_ring_exchange(
                sequence_ids, group=group, participant_ranks=participant_ranks,
                alloc_key=f"{alloc_key}.sequence_ids")
        ),
    )


class _DynamicFlashRingAttention(torch.autograd.Function):
    """Double-buffered async FlashAttention ring on an arbitrary parent-group subset."""

    @staticmethod
    def forward(ctx, q, k, v, positions, sequence_ids, group, participant_ranks, module_id, prefix_metadata, metadata_cache, metadata_cache_key):
        has_sequence_ids = sequence_ids.numel() != 0
        current_k, current_v = k, v
        current_positions = positions
        current_sequence_ids = sequence_ids if has_sequence_ids else None
        running_out = torch.zeros_like(q, dtype=torch.float32)
        running_lse = torch.full(q.shape[:-1], float("-inf"), dtype=torch.float32, device=q.device)
        for step in range(len(participant_ranks)):
            next_exchange = None
            if step + 1 != len(participant_ranks):
                # Two alternating receive buffers let NCCL progress while Flash computes this block.
                next_exchange = _async_subset_block_exchange(
                    current_k, current_v, current_positions, current_sequence_ids,
                    group=group, participant_ranks=participant_ranks,
                    alloc_key=f"hdp.flash_ring.{module_id}.fwd.slot_{step % 2}",
                )
            with torch.profiler.record_function("maltos::hdp.flash.forward.block"):
                block = flash_attn_varlen_prefix_with_lse(
                    q, current_k, current_v,
                    q_positions=positions, k_positions=current_positions,
                    q_sequence_ids=sequence_ids if has_sequence_ids else None,
                    k_sequence_ids=current_sequence_ids, allow_empty_kv=True,
                    metadata=None if prefix_metadata is None else prefix_metadata[step],
                    metadata_cache=metadata_cache,
                    metadata_cache_key=None if metadata_cache_key is None else (*metadata_cache_key, step),
                )
            running_out, running_lse = _merge_flash_attention_blocks(
                running_out, running_lse, block.out, block.lse)
            if step + 1 != len(participant_ranks):
                assert next_exchange is not None
                current_k, current_v, current_positions, current_sequence_ids = next_exchange.wait()
        out = running_out.to(dtype=q.dtype)
        ctx.save_for_backward(q, k, v, out, running_lse, positions, sequence_ids)
        ctx.group, ctx.participant_ranks, ctx.module_id = group, participant_ranks, module_id
        ctx.prefix_metadata = prefix_metadata
        ctx.metadata_cache, ctx.metadata_cache_key = metadata_cache, metadata_cache_key
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, positions, sequence_ids = ctx.saved_tensors
        has_sequence_ids = sequence_ids.numel() != 0
        current_k, current_v = k, v
        current_positions = positions
        current_sequence_ids = sequence_ids if has_sequence_ids else None
        current_dk = torch.zeros_like(k, dtype=torch.float32)
        current_dv = torch.zeros_like(v, dtype=torch.float32)
        dq = torch.zeros_like(q, dtype=torch.float32)
        pending_dk_exchange: _AsyncSubsetExchange | None = None
        pending_dv_exchange: _AsyncSubsetExchange | None = None
        for step in range(len(ctx.participant_ranks)):
            next_block_exchange = None
            if step + 1 != len(ctx.participant_ranks):
                next_block_exchange = _async_subset_block_exchange(
                    current_k, current_v, current_positions, current_sequence_ids,
                    group=ctx.group, participant_ranks=ctx.participant_ranks,
                    alloc_key=f"hdp.flash_ring.{ctx.module_id}.bwd.kv.slot_{step % 2}",
                )
            with torch.profiler.record_function("maltos::hdp.flash.backward.block"):
                block_dq, block_dk, block_dv = flash_attn_varlen_prefix_backward(
                    dout.contiguous(), q, current_k, current_v, out, lse,
                    q_positions=positions, k_positions=current_positions,
                    q_sequence_ids=sequence_ids if has_sequence_ids else None,
                    k_sequence_ids=current_sequence_ids, allow_empty_kv=True,
                    metadata=None if ctx.prefix_metadata is None else ctx.prefix_metadata[step],
                    metadata_cache=ctx.metadata_cache,
                    metadata_cache_key=(
                        None if ctx.metadata_cache_key is None else (*ctx.metadata_cache_key, step)
                    ),
                )
            # The prior owner's gradient exchange progressed while this Flash block ran.
            if pending_dk_exchange is not None:
                current_dk = pending_dk_exchange.wait()
                current_dv = pending_dv_exchange.wait()
            dq += block_dq.float()
            current_dk += block_dk.float()
            current_dv += block_dv.float()
            if step + 1 != len(ctx.participant_ranks):
                pending_dk_exchange = _async_subset_ring_exchange(
                    current_dk, group=ctx.group, participant_ranks=ctx.participant_ranks,
                    alloc_key=f"hdp.flash_ring.{ctx.module_id}.bwd.dk.slot_{step % 2}")
                pending_dv_exchange = _async_subset_ring_exchange(
                    current_dv, group=ctx.group, participant_ranks=ctx.participant_ranks,
                    alloc_key=f"hdp.flash_ring.{ctx.module_id}.bwd.dv.slot_{step % 2}")
                assert next_block_exchange is not None
                current_k, current_v, current_positions, current_sequence_ids = next_block_exchange.wait()
        local_dk = _subset_ring_exchange(current_dk, group=ctx.group, participant_ranks=ctx.participant_ranks)
        local_dv = _subset_ring_exchange(current_dv, group=ctx.group, participant_ranks=ctx.participant_ranks)
        return dq.to(q.dtype), local_dk.to(k.dtype), local_dv.to(v.dtype), None, None, None, None, None, None, None, None


def _merge_flash_attention_blocks(running_out, running_lse, block_out, block_lse):
    merged_lse = torch.logaddexp(running_lse, block_lse)
    previous_scale = torch.where(
        torch.isfinite(running_lse), torch.exp(running_lse - merged_lse), torch.zeros_like(merged_lse))
    block_scale = torch.where(
        torch.isfinite(block_lse), torch.exp(block_lse - merged_lse), torch.zeros_like(merged_lse))
    return running_out * previous_scale.unsqueeze(-1) + block_out.float() * block_scale.unsqueeze(-1), merged_lse


def dynamic_flash_ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    positions: torch.Tensor,
    sequence_ids: torch.Tensor | None,
    group: dist.ProcessGroup,
    participant_ranks: tuple[int, ...],
    module_id: int = 0,
    prefix_metadata: tuple[VarlenPrefixMetadata | None, ...] | None = None,
    metadata_cache: dict[object, VarlenPrefixMetadata] | None = None,
    metadata_cache_key: tuple[object, ...] | None = None,
) -> torch.Tensor:
    """Run custom-autograd Flash Ring without creating a per-document group."""
    if not participant_ranks or len(set(participant_ranks)) != len(participant_ranks):
        raise ValueError("Flash Ring participant_ranks must be a non-empty ordered set")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or k.shape != v.shape:
        raise ValueError("Flash Ring expects Q and native K/V tensors shaped [batch, heads, tokens, head_dim]")
    if q.size(0) != k.size(0) or q.size(2) != k.size(2) or q.size(3) != k.size(3) or q.size(1) % k.size(1):
        raise ValueError("Flash Ring Q heads must be divisible by native KV heads with matching batch/token/head_dim")
    sequence_arg = sequence_ids if sequence_ids is not None else positions.new_empty((0,), dtype=torch.long)
    if prefix_metadata is not None and len(prefix_metadata) != len(participant_ranks):
        raise ValueError("Flash Ring prefix metadata must contain one layout for every ring step")
    return _DynamicFlashRingAttention.apply(
        q, k, v, positions, sequence_arg, group, participant_ranks, module_id,
        prefix_metadata, metadata_cache, metadata_cache_key)
