"""Data-side contract for the CP-only ByteScale HDP loader."""
from __future__ import annotations

import copy

import torch

from data.bytescale_hdp import ByteScaleHdpDataLoader
from parallel.hdp_helper import (
    BYTESCALE_HDP_SCHEDULE_KEY,
    BYTESCALE_HDP_WAVES_KEY,
    ByteScaleHdpBalancedConfig,
)
from data.prefetch import PrefetchDataLoader
from utils.constants import (
    INPUT_IDS_KEY,
    LABELS_KEY,
    POSITION_IDS_KEY,
    SEQUENCE_IDS_KEY,
)


class _CanonicalLoader:
    def __init__(self, batch) -> None:
        self.batch = batch
        self.cursor = 0
        self.consumed_tokens = 0

    def next_batch(self, cp_rank=None):
        assert cp_rank is None
        self.cursor += 1
        self.consumed_tokens += self.batch[INPUT_IDS_KEY].numel()
        return copy.deepcopy(self.batch)

    def state_dict(self):
        return {"cursor": self.cursor, "consumed_tokens": self.consumed_tokens}

    def load_state_dict(self, state) -> None:
        self.cursor = state["cursor"]
        self.consumed_tokens = state["consumed_tokens"]


def main() -> None:
    batch = {
        INPUT_IDS_KEY: torch.arange(10).reshape(1, 10),
        LABELS_KEY: torch.arange(100, 110).reshape(1, 10),
        POSITION_IDS_KEY: torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1]]),
        SEQUENCE_IDS_KEY: torch.tensor([[7, 7, 7, 7, 7, 7, 7, 7, 9, 9]]),
    }
    config = ByteScaleHdpBalancedConfig(6)
    loaders = [
        ByteScaleHdpDataLoader(
            _CanonicalLoader(batch),
            hdp_rank=rank,
            hdp_world_size=2,
            config=config,
        )
        for rank in range(2)
    ]
    local_batches = [loader.next_batch() for loader in loaders]
    local_waves = [batch[BYTESCALE_HDP_WAVES_KEY] for batch in local_batches]
    schedules = [tuple(wave[BYTESCALE_HDP_SCHEDULE_KEY] for wave in waves) for waves in local_waves]

    # The runtime receives only local metadata, but every rank independently
    # derives identical participant membership for their shared long document.
    assert (
        schedules[0][0].documents[0].participant_ranks
        == schedules[1][0].documents[0].participant_ranks
    )
    assert schedules[0][1].document_ids != schedules[1][1].document_ids

    assert all(len(waves) == 2 for waves in local_waves)
    for waves, rank_schedules in zip(local_waves, schedules, strict=True):
        for payload, schedule in zip(waves, rank_schedules, strict=True):
            assert (
                payload[INPUT_IDS_KEY].shape
                == payload[LABELS_KEY].shape
                == payload[POSITION_IDS_KEY].shape
                == payload[SEQUENCE_IDS_KEY].shape
            )
            expected_width = config.partition_tokens if schedule.documents else 1
            assert payload[INPUT_IDS_KEY].shape == (1, expected_width)
            assert schedule.packed_rows <= 1

    state = loaders[0].state_dict()
    before = loaders[0].next_batch()
    loaders[0].load_state_dict(state)
    after = loaders[0].next_batch()
    for before_wave, after_wave in zip(
        before[BYTESCALE_HDP_WAVES_KEY],
        after[BYTESCALE_HDP_WAVES_KEY],
        strict=True,
    ):
        for key in (INPUT_IDS_KEY, LABELS_KEY, POSITION_IDS_KEY, SEQUENCE_IDS_KEY):
            assert torch.equal(before_wave[key], after_wave[key])

    prefetched = PrefetchDataLoader(
        ByteScaleHdpDataLoader(
            _CanonicalLoader(batch),
            hdp_rank=0,
            hdp_world_size=2,
            config=config,
        )
    )
    eager = ByteScaleHdpDataLoader(
        _CanonicalLoader(batch),
        hdp_rank=0,
        hdp_world_size=2,
        config=config,
    )
    try:
        assert torch.equal(
            prefetched.next_batch()[BYTESCALE_HDP_WAVES_KEY][0][INPUT_IDS_KEY],
            eager.next_batch()[BYTESCALE_HDP_WAVES_KEY][0][INPUT_IDS_KEY],
        )
    finally:
        prefetched.close()

    print("ByteScale HDP dataloader PASS")


if __name__ == "__main__":
    main()
