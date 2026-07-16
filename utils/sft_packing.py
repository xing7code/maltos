from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.constants import IGNORE_INDEX, INPUT_IDS_KEY, LABELS_KEY, PAD_SEQUENCE_ID, POSITION_IDS_KEY, SEQUENCE_IDS_KEY
from utils.sft_messages import EncodedSFTExample

INPUT_DTYPE = np.uint32
LABEL_DTYPE = np.int32
POSITION_DTYPE = np.int32
SEQUENCE_DTYPE = np.int32

PACKING_ALGORITHMS = {"next_fit", "best_fit_decreasing"}


@dataclass(frozen=True)
class PackedSFTField:
    dtype: str
    offset: int
    shape: tuple[int, int]

    def to_metadata(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "offset": self.offset,
            "shape": list(self.shape),
        }


@dataclass(frozen=True)
class PackedSFTShard:
    shard_idx: int
    path: str
    sequences: int
    input_field: PackedSFTField
    label_field: PackedSFTField
    position_field: PackedSFTField
    sequence_field: PackedSFTField

    def to_metadata(self) -> dict[str, object]:
        return {
            "shard_idx": self.shard_idx,
            "path": self.path,
            "sequences": self.sequences,
            "fields": {
                INPUT_IDS_KEY: self.input_field.to_metadata(),
                LABELS_KEY: self.label_field.to_metadata(),
                POSITION_IDS_KEY: self.position_field.to_metadata(),
                SEQUENCE_IDS_KEY: self.sequence_field.to_metadata(),
            },
        }


@dataclass(frozen=True)
class PackedSFTSummary:
    raw_tokens: int
    raw_supervised_tokens: int
    packed_sequences: int
    packed_input_tokens: int
    packed_supervised_tokens: int
    padded_tokens: int
    dropped_tail_tokens: int
    shards: list[PackedSFTShard]

    def stats_metadata(self) -> dict[str, object]:
        return {
            "raw_tokens": self.raw_tokens,
            "raw_supervised_tokens": self.raw_supervised_tokens,
            "packed_sequences": self.packed_sequences,
            "packed_input_tokens": self.packed_input_tokens,
            "packed_supervised_tokens": self.packed_supervised_tokens,
            "padded_tokens": self.padded_tokens,
            "dropped_tail_tokens": self.dropped_tail_tokens,
            "num_shards": len(self.shards),
        }


class PackedSFTWriter:
    """Boundary-aware SFT packer with bounded-window bin packing."""

    def __init__(
        self,
        *,
        output_dir: Path,
        seq_len: int,
        sequences_per_shard: int,
        max_sequences: int | None,
        pad_token_id: int,
        packing_algorithm: str = "best_fit_decreasing",
        packing_buffer_size: int = 4096,
    ) -> None:
        if packing_algorithm not in PACKING_ALGORITHMS:
            raise ValueError(
                f"unsupported packing_algorithm={packing_algorithm!r}; "
                f"expected one of {sorted(PACKING_ALGORITHMS)}"
            )
        if packing_buffer_size < 1:
            raise ValueError(f"packing_buffer_size must be >= 1, got {packing_buffer_size}")

        self.output_dir = output_dir
        self.seq_len = seq_len
        self.sequences_per_shard = sequences_per_shard
        self.max_sequences = max_sequences
        self.pad_token_id = pad_token_id
        self.packing_algorithm = packing_algorithm
        self.packing_buffer_size = packing_buffer_size

        self.current_input_ids: list[int] = []
        self.current_labels: list[int] = []
        self.current_position_ids: list[int] = []
        self.current_sequence_ids: list[int] = []
        self.current_sequence_count = 0

        self.pending_segments: list[EncodedSFTExample] = []

        self.shard_inputs: list[list[int]] = []
        self.shard_labels: list[list[int]] = []
        self.shard_position_ids: list[list[int]] = []
        self.shard_sequence_ids: list[list[int]] = []
        self.shards: list[PackedSFTShard] = []
        self.shard_idx = 0

        self.raw_tokens = 0
        self.raw_supervised_tokens = 0
        self.packed_sequences = 0
        self.packed_supervised_tokens = 0
        self.padded_tokens = 0
        self.dropped_tail_tokens = 0
        self.reached_sequence_limit = False

    def add_example(self, example: EncodedSFTExample) -> None:
        if self.reached_sequence_limit:
            return
        if len(example.token_ids) != len(example.supervised_mask):
            raise ValueError("token_ids and supervised_mask must have the same length")
        if len(example.token_ids) < 2:
            return

        self.raw_tokens += len(example.token_ids)
        self.raw_supervised_tokens += sum(int(flag) for flag in example.supervised_mask)

        for segment in split_example_to_seq_len(example, seq_len=self.seq_len):
            if self.reached_sequence_limit:
                return
            if self.packing_algorithm == "next_fit":
                self._add_segment_next_fit(segment)
            else:
                self.pending_segments.append(segment)
                if len(self.pending_segments) >= self.packing_buffer_size:
                    self._flush_pending_segments()

    def finish(self) -> PackedSFTSummary:
        if self.packing_algorithm == "next_fit":
            if self.current_input_ids:
                if self._can_emit_more_sequences():
                    self._emit_current_sequence()
                else:
                    self.dropped_tail_tokens += len(self.current_input_ids)
                    self._reset_current_sequence()
        else:
            self._flush_pending_segments()

        if self.shard_inputs:
            self._flush_shard()

        return PackedSFTSummary(
            raw_tokens=self.raw_tokens,
            raw_supervised_tokens=self.raw_supervised_tokens,
            packed_sequences=self.packed_sequences,
            packed_input_tokens=self.packed_sequences * self.seq_len,
            packed_supervised_tokens=self.packed_supervised_tokens,
            padded_tokens=self.padded_tokens,
            dropped_tail_tokens=self.dropped_tail_tokens,
            shards=list(self.shards),
        )

    def _add_segment_next_fit(self, segment: EncodedSFTExample) -> None:
        if len(segment.token_ids) > self.seq_len:
            raise ValueError(f"segment length exceeds seq_len: {len(segment.token_ids)} > {self.seq_len}")
        if self.current_input_ids and len(self.current_input_ids) + len(segment.token_ids) > self.seq_len:
            self._emit_current_sequence()
        if not self._can_emit_more_sequences():
            self.reached_sequence_limit = True
            self.dropped_tail_tokens += len(segment.token_ids)
            return

        sequence_id = self.current_sequence_count
        for token_idx, token_id in enumerate(segment.token_ids):
            self.current_input_ids.append(int(token_id))
            self.current_position_ids.append(token_idx)
            self.current_sequence_ids.append(sequence_id)
            if token_idx + 1 < len(segment.token_ids) and segment.supervised_mask[token_idx + 1]:
                self.current_labels.append(int(segment.token_ids[token_idx + 1]))
                self.packed_supervised_tokens += 1
            else:
                self.current_labels.append(IGNORE_INDEX)

        self.current_sequence_count += 1
        if len(self.current_input_ids) == self.seq_len:
            self._emit_current_sequence()

    def _flush_pending_segments(self) -> None:
        if not self.pending_segments:
            return

        bins = pack_segments_best_fit_decreasing(self.pending_segments, seq_len=self.seq_len)
        self.pending_segments.clear()

        for bin_idx, segments in enumerate(bins):
            if not self._can_emit_more_sequences():
                self.reached_sequence_limit = True
                self.dropped_tail_tokens += _count_segment_tokens(segments)
                for remaining in bins[bin_idx + 1 :]:
                    self.dropped_tail_tokens += _count_segment_tokens(remaining)
                return
            self._emit_segments_as_sequence(segments)

    def _can_emit_more_sequences(self) -> bool:
        return self.max_sequences is None or self.packed_sequences < self.max_sequences

    def _emit_segments_as_sequence(self, segments: list[EncodedSFTExample]) -> None:
        input_ids: list[int] = []
        labels: list[int] = []
        position_ids: list[int] = []
        sequence_ids: list[int] = []

        for sequence_id, segment in enumerate(segments):
            for token_idx, token_id in enumerate(segment.token_ids):
                input_ids.append(int(token_id))
                position_ids.append(token_idx)
                sequence_ids.append(sequence_id)
                if token_idx + 1 < len(segment.token_ids) and segment.supervised_mask[token_idx + 1]:
                    labels.append(int(segment.token_ids[token_idx + 1]))
                    self.packed_supervised_tokens += 1
                else:
                    labels.append(IGNORE_INDEX)

        if len(input_ids) > self.seq_len:
            raise ValueError(f"packed sequence length exceeds seq_len: {len(input_ids)} > {self.seq_len}")

        pad_len = self.seq_len - len(input_ids)
        input_ids.extend([self.pad_token_id] * pad_len)
        labels.extend([IGNORE_INDEX] * pad_len)
        position_ids.extend([0] * pad_len)
        sequence_ids.extend([PAD_SEQUENCE_ID] * pad_len)
        self.padded_tokens += pad_len

        self.shard_inputs.append(input_ids)
        self.shard_labels.append(labels)
        self.shard_position_ids.append(position_ids)
        self.shard_sequence_ids.append(sequence_ids)
        self.packed_sequences += 1

        if len(self.shard_inputs) >= self.sequences_per_shard:
            self._flush_shard()

    def _emit_current_sequence(self) -> None:
        if not self.current_input_ids:
            return
        if not self._can_emit_more_sequences():
            self.reached_sequence_limit = True
            self.dropped_tail_tokens += len(self.current_input_ids)
            return

        pad_len = self.seq_len - len(self.current_input_ids)
        if pad_len < 0:
            raise ValueError(f"current sequence length exceeds seq_len: {len(self.current_input_ids)} > {self.seq_len}")

        self.current_input_ids.extend([self.pad_token_id] * pad_len)
        self.current_labels.extend([IGNORE_INDEX] * pad_len)
        self.current_position_ids.extend([0] * pad_len)
        self.current_sequence_ids.extend([PAD_SEQUENCE_ID] * pad_len)
        self.padded_tokens += pad_len

        self.shard_inputs.append(list(self.current_input_ids))
        self.shard_labels.append(list(self.current_labels))
        self.shard_position_ids.append(list(self.current_position_ids))
        self.shard_sequence_ids.append(list(self.current_sequence_ids))
        self.packed_sequences += 1

        self._reset_current_sequence()
        if len(self.shard_inputs) >= self.sequences_per_shard:
            self._flush_shard()

    def _reset_current_sequence(self) -> None:
        self.current_input_ids.clear()
        self.current_labels.clear()
        self.current_position_ids.clear()
        self.current_sequence_ids.clear()
        self.current_sequence_count = 0

    def _flush_shard(self) -> None:
        if not self.shard_inputs:
            return

        input_array = np.asarray(self.shard_inputs, dtype=INPUT_DTYPE)
        label_array = np.asarray(self.shard_labels, dtype=LABEL_DTYPE)
        position_array = np.asarray(self.shard_position_ids, dtype=POSITION_DTYPE)
        sequence_array = np.asarray(self.shard_sequence_ids, dtype=SEQUENCE_DTYPE)

        shard_path = self.output_dir / f"shard_{self.shard_idx:05d}.bin"
        rows = int(input_array.shape[0])
        cols = int(input_array.shape[1])
        offset = 0
        input_field = PackedSFTField(dtype=np.dtype(INPUT_DTYPE).name, offset=offset, shape=(rows, cols))
        offset += int(input_array.nbytes)
        label_field = PackedSFTField(dtype=np.dtype(LABEL_DTYPE).name, offset=offset, shape=(rows, cols))
        offset += int(label_array.nbytes)
        position_field = PackedSFTField(dtype=np.dtype(POSITION_DTYPE).name, offset=offset, shape=(rows, cols))
        offset += int(position_array.nbytes)
        sequence_field = PackedSFTField(dtype=np.dtype(SEQUENCE_DTYPE).name, offset=offset, shape=(rows, cols))
        with shard_path.open("wb") as f:
            input_array.tofile(f)
            label_array.tofile(f)
            position_array.tofile(f)
            sequence_array.tofile(f)

        self.shards.append(
            PackedSFTShard(
                shard_idx=self.shard_idx,
                path=shard_path.name,
                sequences=rows,
                input_field=input_field,
                label_field=label_field,
                position_field=position_field,
                sequence_field=sequence_field,
            )
        )
        print(f"wrote shard={self.shard_idx:05d} sequences={input_array.shape[0]:,} path={shard_path.name}")

        self.shard_idx += 1
        self.shard_inputs.clear()
        self.shard_labels.clear()
        self.shard_position_ids.clear()
        self.shard_sequence_ids.clear()


def resolve_pad_token_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    raise ValueError("tokenizer must define pad_token_id or eos_token_id for packed SFT output")


def split_example_to_seq_len(example: EncodedSFTExample, *, seq_len: int) -> list[EncodedSFTExample]:
    if len(example.token_ids) <= seq_len:
        return [example]
    segments: list[EncodedSFTExample] = []
    for start in range(0, len(example.token_ids), seq_len):
        end = min(start + seq_len, len(example.token_ids))
        segments.append(
            EncodedSFTExample(
                token_ids=list(example.token_ids[start:end]),
                supervised_mask=list(example.supervised_mask[start:end]),
            )
        )
    return segments


def pack_segments_best_fit_decreasing(
    segments: list[EncodedSFTExample],
    *,
    seq_len: int,
) -> list[list[EncodedSFTExample]]:
    ordered = sorted(
        enumerate(segments),
        key=lambda item: (-len(item[1].token_ids), item[0]),
    )
    bins: list[list[EncodedSFTExample]] = []
    remaining: list[int] = []

    for _original_idx, segment in ordered:
        seg_len = len(segment.token_ids)
        best_bin_idx = -1
        best_remaining_after = seq_len + 1
        for bin_idx, free_tokens in enumerate(remaining):
            if seg_len > free_tokens:
                continue
            remaining_after = free_tokens - seg_len
            if remaining_after < best_remaining_after:
                best_remaining_after = remaining_after
                best_bin_idx = bin_idx
        if best_bin_idx == -1:
            bins.append([segment])
            remaining.append(seq_len - seg_len)
        else:
            bins[best_bin_idx].append(segment)
            remaining[best_bin_idx] -= seg_len

    return bins


def _count_segment_tokens(segments: list[EncodedSFTExample]) -> int:
    return sum(len(segment.token_ids) for segment in segments)


def export_packing_metadata(
    *,
    summary: PackedSFTSummary,
    seq_len: int,
    sequences_per_shard: int,
    packing_algorithm: str,
    packing_buffer_size: int,
) -> dict[str, object]:
    return {
        "layout": {
            "seq_len": seq_len,
            "row_length": seq_len,
            "record_stride": seq_len,
            "fields": {
                INPUT_IDS_KEY: {"dtype": np.dtype(INPUT_DTYPE).name},
                LABELS_KEY: {"dtype": np.dtype(LABEL_DTYPE).name, "ignore_index": IGNORE_INDEX},
                POSITION_IDS_KEY: {"dtype": np.dtype(POSITION_DTYPE).name},
                SEQUENCE_IDS_KEY: {"dtype": np.dtype(SEQUENCE_DTYPE).name, "pad_sequence_id": PAD_SEQUENCE_ID},
            },
        },
        "strategy": {
            "algorithm": packing_algorithm,
            "buffer_size": packing_buffer_size,
            "name": (
                "example_aware_windowed_best_fit_decreasing"
                if packing_algorithm == "best_fit_decreasing"
                else "example_aware_next_fit"
            ),
            "sequences_per_shard": sequences_per_shard,
        },
        "stats": summary.stats_metadata(),
        "shards": [shard.to_metadata() for shard in summary.shards],
    }
