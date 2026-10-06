"""Rows leave a buffer flush in shuffled order, not longest-first.

Shuffling examples before packing does not fix their order: best-fit-decreasing
re-sorts them by length, so rows arrive ramping from one long document to many
short ones and consecutive optimizer steps see correlated difficulty.
"""
import sys
sys.path.insert(0, ".")
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.sft_messages import EncodedSFTExample as E
from utils.sft_packing import PackedSFTWriter

SEQ, N = 64, 240


def row_head_lengths(seed):
    """Longest document in each emitted row, in emission order.

    Best-fit-decreasing drains the buffer longest-first, so without a shuffle
    this series descends monotonically across the flush.
    """
    with TemporaryDirectory() as td:
        w = PackedSFTWriter(
            output_dir=Path(td), seq_len=SEQ, sequences_per_shard=10_000,
            max_sequences=None, pad_token_id=0,
            packing_algorithm="best_fit_decreasing", packing_buffer_size=N,
            row_shuffle_seed=seed,
        )
        seen = []
        orig = w._emit_segments_as_sequence
        w._emit_segments_as_sequence = lambda segs: (
            seen.append(max(len(s.token_ids) for s in segs)), orig(segs))[1]
        for i in range(N):                       # lengths 4,8,...,  a wide spread
            n = 4 * (i % 15 + 1)
            w.add_example(E([1] * n, [True] * n))
        w.finish()
        return seen


def descending_fraction(xs):
    """Share of adjacent pairs that do not increase - 1.0 means fully sorted."""
    pairs = list(zip(xs, xs[1:]))
    return sum(1 for a, b in pairs if b <= a) / len(pairs)


ordered = row_head_lengths(None)
shuffled = row_head_lengths(1234)
print(f"rows emitted           : {len(ordered)}")
print(f"descending pairs, off  : {descending_fraction(ordered):.3f}")
print(f"descending pairs, on   : {descending_fraction(shuffled):.3f}")
print(f"same seed is stable    : {row_head_lengths(1234) == shuffled}")
print(f"same multiset of rows  : {sorted(ordered) == sorted(shuffled)}")

assert sorted(ordered) == sorted(shuffled), "shuffling must not change what is packed"
assert row_head_lengths(1234) == shuffled, "row shuffle must be reproducible from the seed"
assert descending_fraction(ordered) > 0.9, "expected the unshuffled flush to emit longest-first"
assert descending_fraction(shuffled) < 0.75, "rows still arrive length-ordered"
print("\nPASS")
