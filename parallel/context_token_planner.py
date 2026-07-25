from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

import torch


class ContextTokenPlannerType(str, Enum):
    FIXED_CONTIGUOUS = "fixed_contiguous"
    FIXED_ZIGZAG = "fixed_zigzag"

    @property
    def planner_class(self) -> type["ContextTokenPlanner"]:
        if self is ContextTokenPlannerType.FIXED_CONTIGUOUS:
            return FixedContiguousTokenPlanner
        if self is ContextTokenPlannerType.FIXED_ZIGZAG:
            return FixedZigzagTokenPlanner
        raise ValueError(f"unsupported context token planner type={self!r}")


@dataclass(frozen=True)
class ContextTokenPlan:
    """Ownership of canonical positions for a rectangular CP batch.

    Current fixed CP layouts use a one-dimensional owner vector.  Future
    packed-data planners may return per-token ownership from the batch, but
    the fixed layouts intentionally preserve the existing rectangular local
    tensor contract.
    """

    owner_ranks: torch.Tensor

    def local_positions(self, rank: int) -> torch.Tensor:
        if self.owner_ranks.dim() != 1:
            raise ValueError(
                "fixed CP batch sharding expects one owner rank per sequence position; "
                f"got owner_ranks shape={tuple(self.owner_ranks.shape)}"
            )
        return torch.nonzero(self.owner_ranks == rank, as_tuple=False).flatten()


class ContextTokenPlanner(Protocol):
    planner_type: ContextTokenPlannerType

    def plan(
        self,
        *,
        sequence_lengths: Sequence[int],
        world_size: int,
        device: torch.device | None = None,
    ) -> ContextTokenPlan: ...


class FixedContiguousTokenPlanner:
    """The original equal-size contiguous CP assignment."""

    planner_type = ContextTokenPlannerType.FIXED_CONTIGUOUS

    def plan(
        self,
        *,
        sequence_lengths: Sequence[int],
        world_size: int,
        device: torch.device | None = None,
    ) -> ContextTokenPlan:
        seq_len = _require_rectangular_sequence_lengths(sequence_lengths)
        if seq_len % world_size != 0:
            raise ValueError(
                "ContextParallelPlugin fixed contiguous layout requires sequence length divisible by cp world size, "
                f"got seq_len={seq_len}, cp={world_size}"
            )
        positions = torch.arange(seq_len, dtype=torch.long, device=device)
        return ContextTokenPlan(owner_ranks=positions // (seq_len // world_size))


class FixedZigzagTokenPlanner:
    """The original equal-size zigzag assignment used by Ring CP."""

    planner_type = ContextTokenPlannerType.FIXED_ZIGZAG

    def plan(
        self,
        *,
        sequence_lengths: Sequence[int],
        world_size: int,
        device: torch.device | None = None,
    ) -> ContextTokenPlan:
        seq_len = _require_rectangular_sequence_lengths(sequence_lengths)
        if seq_len % (2 * world_size) != 0:
            raise ValueError(
                "CP ring zigzag requires sequence length divisible by 2 * cp world size, "
                f"got seq_len={seq_len}, cp={world_size}"
            )
        half_len = seq_len // (2 * world_size)
        owners = torch.empty(seq_len, dtype=torch.long, device=device)
        for rank in range(world_size):
            front_start = rank * half_len
            back_start = (2 * world_size - rank - 1) * half_len
            owners[front_start : front_start + half_len] = rank
            owners[back_start : back_start + half_len] = rank
        return ContextTokenPlan(owner_ranks=owners)


def build_context_token_planner(planner_type: ContextTokenPlannerType) -> ContextTokenPlanner:
    """Build the implementation selected by a planner type."""
    return planner_type.planner_class()


def _require_rectangular_sequence_lengths(sequence_lengths: Sequence[int]) -> int:
    """Bridge the legacy rectangular `[batch, seq, ...]` batch contract.

    Fixed layouts own one position vector and apply it identically to every
    batch row.  They therefore intentionally reject variable lengths; dynamic
    planners such as ByteScale/FCP consume the same list-shaped input but will
    return a per-document layout in the next interface phase.
    """
    if not sequence_lengths:
        raise ValueError("Context token planning requires at least one sequence length")
    lengths = tuple(int(length) for length in sequence_lengths)
    if any(length < 1 for length in lengths):
        raise ValueError(f"Context token lengths must be >= 1, got {lengths}")
    if len(set(lengths)) != 1:
        raise ValueError(
            "fixed CP token planners require a rectangular batch with equal sequence lengths, "
            f"got {lengths}"
        )
    return lengths[0]
