"""Autograd-aware point-to-point primitives for context parallelism."""
from __future__ import annotations

import torch
import torch.distributed as dist

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
