"""ByteScale HDP schedule data types shared by data and runtime code."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from utils.constants import INPUT_IDS_KEY, SEQUENCE_IDS_KEY


BYTESCALE_HDP_SCHEDULE_KEY = "bytescale_hdp_schedule"
BYTESCALE_HDP_WAVES_KEY = "bytescale_hdp_waves"


@dataclass(frozen=True)
class ByteScaleHdpBalancedConfig:
    """CP-only ByteScale Algorithm 2 DP-balance reproduction configuration.

    ``partition_tokens`` is the resident-token capacity of one HDP rank.  The
    paper leaves the cost model and several scheduling tie-breaks unspecified.
    This reproduction uses the attention-FLOP proxy ``length ** 2`` and charges
    each selected worker ``length ** 2 / worker_count``.  It makes buckets by
    greedily filling the descending-length sequence order up to total proxy
    FLOPs divided by the HDP degree.  ``balance_delta`` is expressed in that
    same proxy unit and defaults to zero.

    When Algorithm 2's strict target predicate produces no rank, all ranks are
    equally eligible.  When it produces fewer ranks than a sequence needs, the
    remaining workers are the lowest-predicted-time non-target ranks.  Every
    selection is then ordered by ``(predicted_time, rank)``.  These are the
    smallest deterministic completions of details not published in the paper.
    This is not a claim to reproduce ByteScale's unpublished implementation.
    """

    partition_tokens: int
    balance_delta: int = 0

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
