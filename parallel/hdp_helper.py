"""ByteScale HDP schedule data types shared by data and runtime code."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from typing import Any

import torch

from utils.constants import INPUT_IDS_KEY, SEQUENCE_IDS_KEY


BYTESCALE_HDP_SCHEDULE_KEY = "bytescale_hdp_schedule"
BYTESCALE_HDP_WAVES_KEY = "bytescale_hdp_waves"


@dataclass(frozen=True)
class ByteScaleHdpCostModel:
    """Per-sequence execution-time fit obtained from an HDP profiling run.

    ``alpha`` accounts for attention's quadratic work, ``beta`` for token-linear
    work (MLP, projections, norms), and ``gamma`` for each participant's fixed
    scheduling/communication overhead.  Values are deliberately unitless: a
    calibration tool supplies coefficients in a common time unit.
    """

    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0

    def __post_init__(self) -> None:
        if self.alpha < 0 or self.beta < 0 or self.gamma < 0:
            raise ValueError("ByteScale profiling coefficients must be non-negative")

    def sequence_cost(self, sequence_length: int) -> Fraction:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        return (
            Fraction(str(self.alpha)) * sequence_length * sequence_length
            + Fraction(str(self.beta)) * sequence_length
            + Fraction(str(self.gamma))
        )

    def participant_cost(self, sequence_length: int, participant_count: int) -> Fraction:
        if participant_count < 1:
            raise ValueError("participant_count must be positive")
        # The fitted fixed component is paid on every selected HDP rank.
        total_without_fixed = (
            Fraction(str(self.alpha)) * sequence_length * sequence_length
            + Fraction(str(self.beta)) * sequence_length
        )
        return total_without_fixed / participant_count + Fraction(str(self.gamma))


@dataclass(frozen=True)
class ByteScaleHdpBalancedConfig:
    """CP-only ByteScale Algorithm 2 DP-balance reproduction configuration.

    ``partition_tokens`` is the resident-token capacity of one HDP rank.  The
    ``cost_model`` is an explicit profiling fit ``αL² + βL + γ``.  It makes
    buckets from that predicted sequence cost and charges each selected worker
    its sharded quadratic/linear contribution plus its fixed overhead.
    ``balance_delta`` is expressed in the same profiling time unit.

    When Algorithm 2's strict target predicate produces no rank, all ranks are
    equally eligible.  When it produces fewer ranks than a sequence needs, the
    remaining workers are the lowest-predicted-time non-target ranks.  Every
    selection is then ordered by ``(predicted_time, rank)``.  These are the
    smallest deterministic completions of details not published in the paper.
    This is not a claim to reproduce ByteScale's unpublished implementation.
    """

    partition_tokens: int
    balance_delta: float = 0.0
    cost_model: ByteScaleHdpCostModel = field(default_factory=ByteScaleHdpCostModel)

    def __post_init__(self) -> None:
        if self.partition_tokens < 1:
            raise ValueError("ByteScale partition_tokens must be positive")
        if self.partition_tokens % 2:
            raise ValueError(
                "ByteScale partition_tokens must be even for symmetric zigzag"
            )
        if self.balance_delta < 0:
            raise ValueError("ByteScale balance_delta must be non-negative")

    def participant_count(self, sequence_length: int, *, world_size: int) -> int:
        if sequence_length < 1 or world_size < 1:
            raise ValueError("sequence_length and world_size must be positive")
        if sequence_length > world_size * self.partition_tokens:
            raise ValueError(
                "ByteScale CP-only baseline cannot place sequence_length="
                f"{sequence_length} with hdp_world_size={world_size} and "
                f"partition_tokens={self.partition_tokens}; use activation offload "
                "or a larger HDP degree"
            )
        return min(world_size, math.ceil(sequence_length / self.partition_tokens))

    def validate_tp_sp_partition(self, *, tp_size: int, use_sp: bool) -> None:
        if use_sp and self.partition_tokens % tp_size:
            raise ValueError("HDP partition_tokens must be divisible by TP size when SP is enabled")


@dataclass(frozen=True)
class DocumentIndices:
    """One logical document's source row and canonical source columns."""

    source_row: int
    source_indices: tuple[int, ...]

    @classmethod
    def from_batch(cls, batch: dict[str, torch.Tensor]) -> tuple["DocumentIndices", ...]:
        input_ids = batch.get(INPUT_IDS_KEY)
        if not torch.is_tensor(input_ids) or input_ids.dim() != 2:
            raise ValueError("ByteScale requires input_ids [B, S]")
        rows, width = input_ids.shape
        sequence_ids = batch.get(SEQUENCE_IDS_KEY)
        if sequence_ids is None:
            return tuple(cls(row, tuple(range(width))) for row in range(rows))
        if not torch.is_tensor(sequence_ids) or tuple(sequence_ids.shape) != (rows, width):
            raise ValueError("sequence_ids must match input_ids")

        documents: list[DocumentIndices] = []
        for row in range(rows):
            row_ids = sequence_ids[row].detach().cpu().tolist()
            start = 0
            while start < width:
                if int(row_ids[start]) < 0:
                    start += 1
                    continue
                end = start + 1
                while end < width and int(row_ids[end]) == int(row_ids[start]):
                    end += 1
                documents.append(cls(row, tuple(range(start, end))))
                start = end
        if not documents:
            raise ValueError("ByteScale packed batch contains no documents")
        return tuple(documents)


@dataclass(frozen=True)
class ByteScaleHdpLocalDocument:
    document_id: int
    source_row: int
    sequence_length: int
    padded_length: int
    participant_ranks: tuple[int, ...]
    prev_rank: int | None
    next_rank: int | None
    local_row: int
    local_start: int
    local_end: int
    source_indices: tuple[int | None, ...]
    document_positions: tuple[int | None, ...]

    @property
    def local_length(self) -> int:
        return self.local_end - self.local_start


@dataclass(frozen=True)
class ByteScaleHdpLocalLayout:
    rank: int
    hdp_world_size: int
    partition_tokens: int
    scheduler_name: str
    documents: tuple[ByteScaleHdpLocalDocument, ...]
    packed_rows: int
    packed_width: int
    global_valid_targets: int
    global_document_count: int
    wave_index: int = 0
    wave_count: int = 1
    # Runtime-only tensors derived from immutable document placement.  The
    # dictionary is intentionally shared by all attention layers that consume
    # this wave; it is not part of schedule equality or serialization.
    attention_metadata_cache: dict[object, Any] = field(
        default_factory=dict, compare=False, repr=False
    )

    @property
    def document_ids(self) -> tuple[int, ...]:
        return tuple(document.document_id for document in self.documents)

    @property
    def valid_tokens(self) -> int:
        return sum(
            source_index is not None
            for document in self.documents
            for source_index in document.source_indices
        )

    @property
    def placement_tokens(self) -> int:
        return sum(document.local_length for document in self.documents)

    @property
    def local_lengths(self) -> tuple[int, ...]:
        return tuple(document.local_length for document in self.documents)

    @property
    def padded_slots(self) -> int:
        return self.packed_rows * self.packed_width
