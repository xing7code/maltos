"""ByteScale-specific HDP scheduling and rank-local data materialization.

The planner's global placement is transient. The batch contract contains only
the local layout needed by this rank's attention and model forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Sequence

import torch

from parallel.hdp_helper import (
    BYTESCALE_HDP_SCHEDULE_KEY,
    BYTESCALE_HDP_WAVES_KEY,
    ByteScaleHdpBalancedConfig,
    ByteScaleHdpLocalDocument,
    ByteScaleHdpLocalLayout,
    DocumentIndices,
)
from utils.constants import (
    IGNORE_INDEX,
    INPUT_IDS_KEY,
    LABELS_KEY,
    LOSS_WEIGHT_KEY,
    PAD_SEQUENCE_ID,
    POSITION_IDS_KEY,
    SEQUENCE_IDS_KEY,
)


@dataclass(frozen=True)
class _Placement:
    document_id: int
    source: DocumentIndices
    sequence_length: int
    padded_length: int
    participant_ranks: tuple[int, ...]


def schedule_document_waves(
    indices: Sequence[DocumentIndices],
    *,
    rank: int,
    world_size: int,
    config: ByteScaleHdpBalancedConfig,
    global_valid_targets: int,
) -> tuple[ByteScaleHdpLocalLayout, ...]:
    """Plan ordered CP-only DP-balance micro-batch waves.

    The scheduler keeps documents intact, derives their worker count from
    length, processes the paper's descending-length approximately-equal-FLOPs
    buckets in order, and updates predicted worker completion costs after each
    assignment.  See :class:`ByteScaleHdpBalancedConfig` for the explicit
    deterministic completions required by Algorithm 2's unpublished details.
    """
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank} outside world_size={world_size}")

    predicted_times = [Fraction(0) for _ in range(world_size)]
    placement_waves: list[tuple[_Placement, ...]] = []
    ordered_documents = sorted(
        enumerate(indices),
        key=lambda item: (-len(item[1].source_indices), item[0]),
    )
    for bucket in _bucket_by_attention_flops(ordered_documents, world_size):
        remaining = list(bucket)
        while remaining:
            remaining_capacity = [config.partition_tokens] * world_size
            wave: list[_Placement] = []
            while remaining:
                document_id, source = remaining[0]
                sequence_length = len(source.source_indices)
                degree = config.participant_count(sequence_length, world_size=world_size)
                padded_length = _round_up(sequence_length, 2 * degree)
                local_length = padded_length // degree
                participants = _select_algorithm_2_workers(
                    predicted_times,
                    worker_count=degree,
                    balance_delta=config.balance_delta,
                    eligible_ranks=tuple(
                        candidate
                        for candidate, capacity in enumerate(remaining_capacity)
                        if capacity >= local_length
                    ),
                )
                if participants is None:
                    break
                placement = _Placement(
                    document_id,
                    source,
                    sequence_length,
                    padded_length,
                    participants,
                )
                wave.append(placement)
                remaining.pop(0)
                for participant in participants:
                    remaining_capacity[participant] -= local_length
                    predicted_times[participant] += Fraction(
                        sequence_length * sequence_length,
                        degree,
                    )
            if not wave:
                raise AssertionError("validated document could not fit an empty HDP wave")
            placement_waves.append(tuple(wave))

    return tuple(
        _local_layout_for_wave(
            placements,
            rank=rank,
            world_size=world_size,
            config=config,
            global_valid_targets=global_valid_targets,
            global_document_count=len(indices),
            wave_index=wave_index,
            wave_count=len(placement_waves),
        )
        for wave_index, placements in enumerate(placement_waves)
    )


def schedule_documents(
    indices: Sequence[DocumentIndices],
    *,
    rank: int,
    world_size: int,
    config: ByteScaleHdpBalancedConfig,
    global_valid_targets: int,
) -> ByteScaleHdpLocalLayout:
    """Return the sole wave, rejecting schedules that need multiple waves.

    Runtime training consumes :func:`schedule_document_waves` through the HDP
    dataloader envelope. This compatibility helper deliberately refuses to
    collapse Algorithm 2's micro-batch sequence into an oversized tensor.
    """
    layouts = schedule_document_waves(
        indices,
        rank=rank,
        world_size=world_size,
        config=config,
        global_valid_targets=global_valid_targets,
    )
    if len(layouts) != 1:
        raise ValueError(
            "ByteScale schedule contains multiple waves; use the HDP dataloader "
            "envelope instead of schedule_documents()"
        )
    return layouts[0]


def _local_layout_for_wave(
    placements: Sequence[_Placement],
    *,
    rank: int,
    world_size: int,
    config: ByteScaleHdpBalancedConfig,
    global_valid_targets: int,
    global_document_count: int,
    wave_index: int,
    wave_count: int,
) -> ByteScaleHdpLocalLayout:

    row_remaining: list[int] = []
    local_documents: list[ByteScaleHdpLocalDocument] = []
    for placement in placements:
        if rank not in placement.participant_ranks:
            continue

        local_rank = placement.participant_ranks.index(rank)
        local_length = placement.padded_length // len(placement.participant_ranks)
        row, start = _best_fit_row(
            row_remaining,
            local_length,
            config.partition_tokens,
        )
        if row == len(row_remaining):
            row_remaining.append(config.partition_tokens)
        row_remaining[row] -= local_length

        canonical_positions = _zigzag_positions(
            placement.padded_length,
            len(placement.participant_ranks),
            local_rank,
        )
        source_indices = tuple(
            placement.source.source_indices[position]
            if position < placement.sequence_length
            else None
            for position in canonical_positions
        )
        document_positions = tuple(
            position if position < placement.sequence_length else None
            for position in canonical_positions
        )
        participants = placement.participant_ranks
        local_documents.append(
            ByteScaleHdpLocalDocument(
                document_id=placement.document_id,
                source_row=placement.source.source_row,
                sequence_length=placement.sequence_length,
                padded_length=placement.padded_length,
                participant_ranks=participants,
                prev_rank=(
                    participants[(local_rank - 1) % len(participants)]
                    if len(participants) > 1
                    else None
                ),
                next_rank=(
                    participants[(local_rank + 1) % len(participants)]
                    if len(participants) > 1
                    else None
                ),
                local_row=row,
                local_start=start,
                local_end=start + local_length,
                source_indices=source_indices,
                document_positions=document_positions,
            )
        )

    if local_documents and len(row_remaining) != 1:
        raise AssertionError("HDP wave packing exceeded its resident-token capacity")

    return ByteScaleHdpLocalLayout(
        rank=rank,
        hdp_world_size=world_size,
        partition_tokens=config.partition_tokens,
        scheduler_name="bytescale_dp_balance_reproduction",
        documents=tuple(local_documents),
        packed_rows=len(row_remaining),
        packed_width=config.partition_tokens,
        global_valid_targets=global_valid_targets,
        global_document_count=global_document_count,
        wave_index=wave_index,
        wave_count=wave_count,
    )


def _bucket_by_attention_flops(
    ordered_documents: Sequence[tuple[int, DocumentIndices]],
    world_size: int,
) -> tuple[tuple[tuple[int, DocumentIndices], ...], ...]:
    """Construct Algorithm 2's ordered approximately-equal-FLOPs buckets.

    The paper does not define bucket count or a boundary policy.  We use one
    rank-share of total ``length ** 2`` proxy FLOPs as the deterministic target
    and never split a document: append while it fits, otherwise start the next
    bucket.  An oversized document is therefore its own bucket.
    """
    total_flops = sum(len(source.source_indices) ** 2 for _, source in ordered_documents)
    target_flops = max(1, math.ceil(total_flops / world_size))
    buckets: list[tuple[tuple[int, DocumentIndices], ...]] = []
    current: list[tuple[int, DocumentIndices]] = []
    current_flops = 0
    for item in ordered_documents:
        flops = len(item[1].source_indices) ** 2
        if current and current_flops + flops > target_flops:
            buckets.append(tuple(current))
            current = []
            current_flops = 0
        current.append(item)
        current_flops += flops
    if current:
        buckets.append(tuple(current))
    return tuple(buckets)


def _select_algorithm_2_workers(
    predicted_times: Sequence[Fraction],
    *,
    worker_count: int,
    balance_delta: int,
    eligible_ranks: Sequence[int],
) -> tuple[int, ...] | None:
    """Choose one CP worker group from Algorithm 2's target ranks.

    ``target_ranks`` is exactly Algorithm 2's strict
    ``max_time - exec_time > delta`` set.  Its empty/too-small cases are
    completed using the documented deterministic rule in the configuration.
    """
    max_time = max(predicted_times, default=Fraction(0))
    eligible = set(eligible_ranks)
    if len(eligible) < worker_count:
        return None
    target_ranks = [
        rank
        for rank, predicted_time in enumerate(predicted_times)
        if rank in eligible and max_time - predicted_time > balance_delta
    ]
    ordered_targets = sorted(target_ranks, key=lambda rank: (predicted_times[rank], rank))
    if not ordered_targets:
        ordered_targets = sorted(
            eligible,
            key=lambda rank: (predicted_times[rank], rank),
        )
    if len(ordered_targets) < worker_count:
        remaining_ranks = sorted(
            (rank for rank in eligible if rank not in target_ranks),
            key=lambda rank: (predicted_times[rank], rank),
        )
        ordered_targets.extend(remaining_ranks)
    return tuple(sorted(ordered_targets[:worker_count]))


class ByteScaleHdpDataLoader:
    """Wrap a canonical loader and materialize one HDP rank's packed batch."""

    def __init__(
        self,
        loader: Any,
        *,
        hdp_rank: int,
        hdp_world_size: int,
        config: ByteScaleHdpBalancedConfig,
    ) -> None:
        self.loader = loader
        self.hdp_rank = hdp_rank
        self.hdp_world_size = hdp_world_size
        self.config = config

    @property
    def consumed_tokens(self):
        return self.loader.consumed_tokens

    def next_batch(self, cp_rank: int | None = None):
        if cp_rank is not None:
            raise ValueError("ByteScale HDP is on DP; cp_rank must be None")
        waves, _ = build_bytescale_local_batches(
            self.loader.next_batch(None),
            rank=self.hdp_rank,
            world_size=self.hdp_world_size,
            config=self.config,
        )
        return {BYTESCALE_HDP_WAVES_KEY: waves}

    def state_dict(self):
        return self.loader.state_dict()

    def load_state_dict(self, state):
        return self.loader.load_state_dict(state)

    def close(self) -> None:
        close = getattr(self.loader, "close", None)
        if close is not None:
            close()


def build_bytescale_local_batch(
    batch: dict[str, torch.Tensor],
    *,
    rank: int,
    world_size: int,
    config: ByteScaleHdpBalancedConfig,
) -> tuple[dict[str, Any], ByteScaleHdpLocalLayout]:
    waves, layouts = build_bytescale_local_batches(
        batch,
        rank=rank,
        world_size=world_size,
        config=config,
    )
    if len(waves) != 1:
        raise ValueError(
            "ByteScale schedule contains multiple waves; use "
            "build_bytescale_local_batches() or ByteScaleHdpDataLoader"
        )
    return waves[0], layouts[0]


def build_bytescale_local_batches(
    batch: dict[str, torch.Tensor],
    *,
    rank: int,
    world_size: int,
    config: ByteScaleHdpBalancedConfig,
) -> tuple[tuple[dict[str, Any], ...], tuple[ByteScaleHdpLocalLayout, ...]]:
    indices = DocumentIndices.from_batch(batch)
    labels = batch.get(LABELS_KEY)
    global_valid_targets = (
        int((labels != IGNORE_INDEX).sum().item()) if torch.is_tensor(labels) else 0
    )
    layouts = schedule_document_waves(
        indices,
        rank=rank,
        world_size=world_size,
        config=config,
        global_valid_targets=global_valid_targets,
    )
    return tuple(
        _materialize_bytescale_local_wave(batch, layout, world_size=world_size)
        for layout in layouts
    ), layouts


def _materialize_bytescale_local_wave(
    batch: dict[str, torch.Tensor],
    layout: ByteScaleHdpLocalLayout,
    *,
    world_size: int,
) -> dict[str, Any]:
    if not layout.documents:
        return _dummy_batch(batch, layout)

    source = batch[INPUT_IDS_KEY]
    labels = batch.get(LABELS_KEY)
    result: dict[str, Any] = {
        INPUT_IDS_KEY: torch.zeros(
            (layout.packed_rows, layout.packed_width),
            dtype=source.dtype,
            device=source.device,
        ),
        LABELS_KEY: torch.full(
            (layout.packed_rows, layout.packed_width),
            IGNORE_INDEX,
            dtype=torch.long,
            device=source.device,
        ),
        POSITION_IDS_KEY: torch.zeros(
            (layout.packed_rows, layout.packed_width),
            dtype=torch.long,
            device=source.device,
        ),
        SEQUENCE_IDS_KEY: torch.full(
            (layout.packed_rows, layout.packed_width),
            PAD_SEQUENCE_ID,
            dtype=torch.long,
            device=source.device,
        ),
    }

    positions = batch.get(POSITION_IDS_KEY)
    for document in layout.documents:
        for offset, (source_index, position) in enumerate(
            zip(document.source_indices, document.document_positions, strict=True)
        ):
            if source_index is None:
                continue
            row = document.local_row
            column = document.local_start + offset
            result[INPUT_IDS_KEY][row, column] = source[document.source_row, source_index]
            if torch.is_tensor(labels):
                result[LABELS_KEY][row, column] = labels[document.source_row, source_index]
            result[POSITION_IDS_KEY][row, column] = (
                positions[document.source_row, source_index]
                if torch.is_tensor(positions)
                else position
            )
            result[SEQUENCE_IDS_KEY][row, column] = document.document_id

    local_valid_targets = int((result[LABELS_KEY] != IGNORE_INDEX).sum().item())
    result[LOSS_WEIGHT_KEY] = (
        world_size * local_valid_targets / layout.global_valid_targets
        if layout.global_valid_targets
        else 0.0
    )
    result[BYTESCALE_HDP_SCHEDULE_KEY] = layout
    return result


def _dummy_batch(
    source_batch: dict[str, torch.Tensor],
    layout: ByteScaleHdpLocalLayout,
) -> dict[str, Any]:
    source = source_batch[INPUT_IDS_KEY]
    return {
        INPUT_IDS_KEY: torch.zeros((1, 1), dtype=source.dtype, device=source.device),
        LABELS_KEY: torch.full(
            (1, 1),
            IGNORE_INDEX,
            dtype=torch.long,
            device=source.device,
        ),
        POSITION_IDS_KEY: torch.zeros((1, 1), dtype=torch.long, device=source.device),
        SEQUENCE_IDS_KEY: torch.full(
            (1, 1),
            PAD_SEQUENCE_ID,
            dtype=torch.long,
            device=source.device,
        ),
        LOSS_WEIGHT_KEY: 0.0,
        BYTESCALE_HDP_SCHEDULE_KEY: layout,
    }


def _best_fit_row(
    remaining: Sequence[int],
    length: int,
    capacity: int,
) -> tuple[int, int]:
    candidates = [
        (space, row)
        for row, space in enumerate(remaining)
        if space >= length
    ]
    if not candidates:
        return len(remaining), 0
    space, row = min(candidates)
    return row, capacity - space


def _zigzag_positions(
    padded_length: int,
    degree: int,
    local_rank: int,
) -> tuple[int, ...]:
    chunk = padded_length // (2 * degree)
    return tuple(range(local_rank * chunk, (local_rank + 1) * chunk)) + tuple(
        range(
            (2 * degree - local_rank - 1) * chunk,
            (2 * degree - local_rank) * chunk,
        )
    )


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
