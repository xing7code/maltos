from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from data import PrefetchDataLoader, PretrainingDataLoader, TokenShardDataset
from parallel.context_batch import ContextParallelBatchSharder
from parallel.context_token_planner import FixedContiguousTokenPlanner, FixedZigzagTokenPlanner
from utils.constants import INPUT_IDS_KEY, LABELS_KEY, POSITION_IDS_KEY


def _assert_same_batch(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        if torch.is_tensor(actual[key]):
            assert torch.equal(actual[key], expected[key]), key
        else:
            assert actual[key] == expected[key], key


def test_loader_can_return_canonical_or_runtime_equivalent_cp_local_batch() -> None:
    with tempfile.TemporaryDirectory(prefix="maltos_cp_loader_") as root:
        path = Path(root) / "tokens.bin"
        np.arange(256, dtype=np.uint32).tofile(path)
        sharder = ContextParallelBatchSharder(FixedZigzagTokenPlanner(), world_size=2)

        canonical_loader = PretrainingDataLoader(
            TokenShardDataset([path]), seq_len=8, micro_batch_size=2
        )
        local_loader = PretrainingDataLoader(
            TokenShardDataset([path]), seq_len=8, micro_batch_size=2, cp_batch_sharder=sharder
        )

        canonical = canonical_loader.next_batch()
        local = local_loader.next_batch(1)
        _assert_same_batch(local, sharder.shard(canonical, rank=1))
        assert canonical[INPUT_IDS_KEY].shape == (2, 8)
        assert local[INPUT_IDS_KEY].shape == (2, 4)
        assert local[POSITION_IDS_KEY].tolist() == [[2, 3, 4, 5], [2, 3, 4, 5]]


def test_prefetch_preserves_cp_local_layout() -> None:
    with tempfile.TemporaryDirectory(prefix="maltos_cp_prefetch_") as root:
        path = Path(root) / "tokens.bin"
        np.arange(256, dtype=np.uint32).tofile(path)
        sharder = ContextParallelBatchSharder(FixedContiguousTokenPlanner(), world_size=2)

        direct = PretrainingDataLoader(
            TokenShardDataset([path]), seq_len=8, micro_batch_size=1, cp_batch_sharder=sharder
        )
        prefetched = PrefetchDataLoader(
            PretrainingDataLoader(
                TokenShardDataset([path]), seq_len=8, micro_batch_size=1, cp_batch_sharder=sharder
            )
        )
        try:
            for _ in range(3):
                _assert_same_batch(prefetched.next_batch(1), direct.next_batch(1))
        finally:
            prefetched.close()


def test_cp_rank_requires_runtime_injected_layout() -> None:
    with tempfile.TemporaryDirectory(prefix="maltos_cp_loader_") as root:
        path = Path(root) / "tokens.bin"
        np.arange(32, dtype=np.uint32).tofile(path)
        loader = PretrainingDataLoader(TokenShardDataset([path]), seq_len=8, micro_batch_size=1)
        try:
            loader.next_batch(0)
        except RuntimeError as error:
            assert "constructor cp_batch_sharder" in str(error)
        else:
            raise AssertionError("CP-local batches must use an injected runtime layout")


def main() -> None:
    test_loader_can_return_canonical_or_runtime_equivalent_cp_local_batch()
    test_prefetch_preserves_cp_local_layout()
    test_cp_rank_requires_runtime_injected_layout()
    print("context-parallel dataloader PASS")


if __name__ == "__main__":
    main()
