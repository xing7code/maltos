from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import torch


class ContextTokenPlannerPhase(str, Enum):
    """Runtime boundary at which a CP token plan is applied."""

    PRE_STEP_RUNNER = "pre_step_runner"
    PRE_FORWARD = "pre_forward"


class ContextTokenPlannerType(str, Enum):
    FIXED_CONTIGUOUS = "fixed_contiguous"
    FIXED_ZIGZAG = "fixed_zigzag"


@dataclass(frozen=True)
class ContextTokenPlannerConfig:
    """Placement lifetime policy shared by CP token planners.

    ``restore_phase`` is deliberately a string instead of a runtime enum: the
    parallel package must not depend on the runtime package.  It will be used
    by dynamic planners that need canonical output ordering.
    """

    planner_type: ContextTokenPlannerType = ContextTokenPlannerType.FIXED_CONTIGUOUS
    plan_phase: ContextTokenPlannerPhase = ContextTokenPlannerPhase.PRE_STEP_RUNNER
    restore_phase: str | None = None


@dataclass(frozen=True)
class ContextTokenPlan:
    """Ownership of canonical sequence positions for one CP group.

    Current fixed CP layouts use a one-dimensional owner vector.  Future
    packed-data planners may return per-token ownership from the batch, but
    the fixed layouts intentionally preserve the existing rectangular local
    tensor contract.
    """

    owner_ranks: torch.Tensor
    restore_indices: torch.Tensor | None = None

    def local_positions(self, rank: int) -> torch.Tensor:
        if self.owner_ranks.dim() != 1:
            raise ValueError(
                "fixed CP batch sharding expects one owner rank per sequence position; "
                f"got owner_ranks shape={tuple(self.owner_ranks.shape)}"
            )
        return torch.nonzero(self.owner_ranks == rank, as_tuple=False).flatten()


class ContextTokenPlanner(Protocol):
    config: ContextTokenPlannerConfig

    def plan(self, *, seq_len: int, world_size: int, device: torch.device | None = None) -> ContextTokenPlan: ...

    def restore(self, outputs: Any, plan: ContextTokenPlan) -> Any: ...


@dataclass(frozen=True)
class FixedContiguousTokenPlanner:
    """The original equal-size contiguous CP assignment."""

    config: ContextTokenPlannerConfig = ContextTokenPlannerConfig()

    def plan(self, *, seq_len: int, world_size: int, device: torch.device | None = None) -> ContextTokenPlan:
        if seq_len % world_size != 0:
            raise ValueError(
                "ContextParallelPlugin fixed contiguous layout requires sequence length divisible by cp world size, "
                f"got seq_len={seq_len}, cp={world_size}"
            )
        positions = torch.arange(seq_len, dtype=torch.long, device=device)
        return ContextTokenPlan(owner_ranks=positions // (seq_len // world_size))

    def restore(self, outputs: Any, plan: ContextTokenPlan) -> Any:
        return outputs


@dataclass(frozen=True)
class FixedZigzagTokenPlanner:
    """The original equal-size zigzag assignment used by Ring CP."""

    config: ContextTokenPlannerConfig = ContextTokenPlannerConfig(
        planner_type=ContextTokenPlannerType.FIXED_ZIGZAG,
    )

    def plan(self, *, seq_len: int, world_size: int, device: torch.device | None = None) -> ContextTokenPlan:
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

    def restore(self, outputs: Any, plan: ContextTokenPlan) -> Any:
        return outputs


def build_context_token_planner(config: ContextTokenPlannerConfig) -> ContextTokenPlanner:
    """Build a placement policy independently of the CP attention core."""
    if config.planner_type == ContextTokenPlannerType.FIXED_CONTIGUOUS:
        return FixedContiguousTokenPlanner(config=config)
    if config.planner_type == ContextTokenPlannerType.FIXED_ZIGZAG:
        return FixedZigzagTokenPlanner(config=config)
    raise ValueError(f"unsupported context token planner type={config.planner_type!r}")
