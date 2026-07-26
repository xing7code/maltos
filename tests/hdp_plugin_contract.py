from __future__ import annotations

from types import SimpleNamespace

import torch

from data.bytescale_hdp import (
    build_bytescale_local_batch,
    build_bytescale_local_batches,
    schedule_document_waves,
)
from parallel.hdp_helper import ByteScaleHdpBalancedConfig, DocumentIndices
from runtime.mesh import MeshAxis
from runtime.plugin import PluginId
from runtime.plugins.cp import ContextParallelPlugin
from runtime.plugins.hdp import ByteScaleHdpPlugin, _layout_metrics
from utils.constants import (
    IGNORE_INDEX,
    INPUT_IDS_KEY,
    LABELS_KEY,
    POSITION_IDS_KEY,
    SEQUENCE_IDS_KEY,
)


class _FakeGroup:
    pass


_HDP_GROUP = _FakeGroup()


def _patch_dist(world_size: int = 4) -> None:
    return None


def _config() -> ByteScaleHdpBalancedConfig:
    return ByteScaleHdpBalancedConfig(partition_tokens=8)


def test_hdp_balanced_participant_count_is_proportional_to_length() -> None:
    config = _config()
    assert config.participant_count(8, world_size=4) == 1
    assert config.participant_count(24, world_size=4) == 3
    assert config.participant_count(32, world_size=4) == 4


def test_hdp_balanced_partition_tokens_must_be_even_for_zigzag() -> None:
    try:
        ByteScaleHdpBalancedConfig(partition_tokens=7)
    except ValueError as exc:
        assert "must be even" in str(exc)
    else:
        raise AssertionError("HDP zigzag must reject an odd partition length")


def test_hdp_rejects_sequences_beyond_cp_only_capacity() -> None:
    try:
        ByteScaleHdpBalancedConfig(partition_tokens=4).participant_count(
            17,
            world_size=2,
        )
    except ValueError as exc:
        assert "activation offload" in str(exc)
    else:
        raise AssertionError("CP-only HDP must reject a sequence beyond capacity")


def test_scheduler_expresses_multiple_ordered_waves() -> None:
    indices = tuple(DocumentIndices(0, tuple(range(4))) for _ in range(3))
    config = ByteScaleHdpBalancedConfig(partition_tokens=4)
    layouts = tuple(
        schedule_document_waves(
            indices,
            rank=rank,
            world_size=2,
            config=config,
            global_valid_targets=12,
        )
        for rank in range(2)
    )
    assert all(len(rank_layouts) == 3 for rank_layouts in layouts)
    assert tuple(layout.wave_index for layout in layouts[0]) == (0, 1, 2)
    assert all(layout.packed_rows <= 1 for rank_layouts in layouts for layout in rank_layouts)


def _participant_groups(
    indices: tuple[DocumentIndices, ...],
    *,
    world_size: int,
    config: ByteScaleHdpBalancedConfig,
) -> tuple[tuple[int, ...], ...]:
    groups: dict[int, tuple[int, ...]] = {}
    for rank in range(world_size):
        layouts = schedule_document_waves(
            indices,
            rank=rank,
            world_size=world_size,
            config=config,
            global_valid_targets=sum(len(index.source_indices) for index in indices),
        )
        for layout in layouts:
            assert layout.scheduler_name == "bytescale_dp_balance_reproduction"
            for document in layout.documents:
                groups[document.document_id] = document.participant_ranks
    return tuple(groups[document_id] for document_id in range(len(indices)))


def test_algorithm_2_dp_balance_uses_hand_computed_buckets_and_costs() -> None:
    # L² proxy FLOPs are [256, 64, 64, 16, 16], so with four ranks the
    # deterministic 104-FLOP buckets are [16], [8], and [8, 4, 4].
    # Worker costs are L² / D(L): 128 for the 16-token two-rank document,
    # then 64, 64, 16, and 16.  Algorithm 2 target-rank updates yield:
    # (0, 1), (2,), (3,), (2,), (3,).
    indices = tuple(
        DocumentIndices(0, tuple(range(length)))
        for length in (16, 8, 8, 4, 4)
    )
    groups = _participant_groups(
        indices,
        world_size=4,
        config=ByteScaleHdpBalancedConfig(partition_tokens=12),
    )
    assert groups == ((0, 1), (2,), (3,), (2,), (3,))


def test_algorithm_2_ties_break_by_rank_then_document_order() -> None:
    indices = tuple(DocumentIndices(0, tuple(range(8))) for _ in range(4))
    groups = _participant_groups(
        indices,
        world_size=4,
        config=ByteScaleHdpBalancedConfig(partition_tokens=8),
    )
    assert groups == ((0,), (1,), (2,), (3,))
    assert groups == _participant_groups(
        indices,
        world_size=4,
        config=ByteScaleHdpBalancedConfig(partition_tokens=8),
    )


def test_algorithm_2_can_leave_idle_ranks_with_an_empty_local_layout() -> None:
    indices = (DocumentIndices(0, tuple(range(4))),)
    layouts = tuple(
        schedule_document_waves(
            indices,
            rank=rank,
            world_size=4,
            config=ByteScaleHdpBalancedConfig(partition_tokens=4),
            global_valid_targets=4,
        )
        for rank in range(4)
    )
    assert layouts[0][0].document_ids == (0,)
    assert all(rank_layouts[0].document_ids == () for rank_layouts in layouts[1:])


def test_plugin_rejects_fixed_cp() -> None:
    _patch_dist()
    hdp = ByteScaleHdpPlugin(config=_config())
    runtime = SimpleNamespace(
        mesh=SimpleNamespace(dp=2, cp=2),
        plugins=[hdp, ContextParallelPlugin()],
    )
    try:
        hdp.bind(runtime)
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("ByteScale HDP must not coexist with fixed CP")


def test_plugin_rejects_pipeline_parallelism() -> None:
    hdp = ByteScaleHdpPlugin(config=_config())
    runtime = SimpleNamespace(
        mesh=SimpleNamespace(dp=2, cp=1, pp=2),
        plugins=[hdp],
    )
    try:
        hdp.bind(runtime)
    except ValueError as exc:
        assert "pp_size=1" in str(exc)
    else:
        raise AssertionError("CP-only ByteScale HDP must reject PP")


def test_plugin_id_is_hdp() -> None:
    assert ByteScaleHdpPlugin(config=_config()).id is PluginId.HDP


def test_hdp_balanced_layout_is_proportional_and_zigzag() -> None:
    indices = tuple(DocumentIndices(0, tuple(range(length))) for length in (24, 8))
    layout = schedule_document_waves(
        indices,
        rank=0,
        world_size=4,
        config=_config(),
        global_valid_targets=32,
    )
    long_document = layout[0].documents[0]
    assert long_document.participant_ranks == (0, 1, 2)
    assert long_document.padded_length == 24
    # The 2D symmetric layout balances causal attention work within the
    # document's three-worker partition.
    assert long_document.source_indices == (0, 1, 2, 3, 20, 21, 22, 23)
    assert layout[0].document_ids == (0,)
    metrics = _layout_metrics(layout[0])
    assert metrics["local_attention_work"] == 192


def test_packed_batch_materialization_preserves_zigzag_and_masks_padding() -> None:
    batch = {
        INPUT_IDS_KEY: torch.arange(24, dtype=torch.long).reshape(1, 24),
        LABELS_KEY: torch.arange(100, 124, dtype=torch.long).reshape(1, 24),
        POSITION_IDS_KEY: torch.arange(24, dtype=torch.long).reshape(1, 24),
        SEQUENCE_IDS_KEY: torch.zeros((1, 24), dtype=torch.long),
    }
    local, layout = build_bytescale_local_batch(
        batch,
        rank=0,
        world_size=4,
        config=_config(),
    )
    assert layout.document_ids == (0,)
    assert layout.local_lengths == (8,)
    assert local[INPUT_IDS_KEY].tolist() == [[0, 1, 2, 3, 20, 21, 22, 23]]
    assert local[POSITION_IDS_KEY].tolist() == local[INPUT_IDS_KEY].tolist()
    assert local[SEQUENCE_IDS_KEY].tolist() == [[0] * 8]
    assert not bool((local[LABELS_KEY] == IGNORE_INDEX).any())


def test_packed_position_fallback_is_document_relative() -> None:
    batch = {
        INPUT_IDS_KEY: torch.tensor([[10, 11, 12, 13]]),
        LABELS_KEY: torch.tensor([[10, 11, 12, 13]]),
        SEQUENCE_IDS_KEY: torch.tensor([[0, 0, 1, 1]]),
    }
    local, layout = build_bytescale_local_batch(
        batch,
        rank=0,
        world_size=1,
        config=ByteScaleHdpBalancedConfig(partition_tokens=4),
    )
    assert layout.document_ids == (0, 1)
    assert local[POSITION_IDS_KEY].tolist() == [[0, 1, 0, 1]]


def test_best_fit_packing_tracks_valid_and_padded_slots() -> None:
    batch = {
        INPUT_IDS_KEY: torch.arange(10).reshape(1, 10),
        LABELS_KEY: torch.arange(10).reshape(1, 10),
        SEQUENCE_IDS_KEY: torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]),
    }
    waves, layouts = build_bytescale_local_batches(
        batch,
        rank=0,
        world_size=2,
        config=ByteScaleHdpBalancedConfig(partition_tokens=6),
    )
    assert len(waves) == len(layouts) == 2
    assert all(layout.packed_rows <= 1 for layout in layouts)
    assert layouts[0].placement_tokens == 4
    assert layouts[1].placement_tokens == 2
    assert waves[0][SEQUENCE_IDS_KEY].tolist() == [[0, 0, 0, 0, -1, -1]]
    assert waves[1][SEQUENCE_IDS_KEY].tolist() == [[1, 1, -1, -1, -1, -1]]


def test_repeated_source_sequence_ids_get_unique_document_identity() -> None:
    batch = {
        INPUT_IDS_KEY: torch.tensor([[1, 2, 0, 0], [3, 4, 0, 0]]),
        LABELS_KEY: torch.tensor([[1, 2, IGNORE_INDEX, IGNORE_INDEX], [3, 4, IGNORE_INDEX, IGNORE_INDEX]]),
        SEQUENCE_IDS_KEY: torch.tensor([[0, 0, -1, -1], [0, 0, -1, -1]]),
    }
    local, layout = build_bytescale_local_batch(
        batch,
        rank=0,
        world_size=1,
        config=ByteScaleHdpBalancedConfig(partition_tokens=4),
    )
    assert layout.document_ids == (0, 1)
    assert local[SEQUENCE_IDS_KEY].tolist() == [[0, 0, 1, 1]]
    assert local[POSITION_IDS_KEY].tolist() == [[0, 1, 0, 1]]


def main() -> None:
    test_hdp_balanced_participant_count_is_proportional_to_length()
    test_hdp_balanced_partition_tokens_must_be_even_for_zigzag()
    test_hdp_rejects_sequences_beyond_cp_only_capacity()
    test_scheduler_expresses_multiple_ordered_waves()
    test_algorithm_2_dp_balance_uses_hand_computed_buckets_and_costs()
    test_algorithm_2_ties_break_by_rank_then_document_order()
    test_algorithm_2_can_leave_idle_ranks_with_an_empty_local_layout()
    test_plugin_rejects_fixed_cp()
    test_plugin_rejects_pipeline_parallelism()
    test_plugin_id_is_hdp()
    test_hdp_balanced_layout_is_proportional_and_zigzag()
    test_packed_batch_materialization_preserves_zigzag_and_masks_padding()
    test_packed_position_fallback_is_document_relative()
    test_best_fit_packing_tracks_valid_and_padded_slots()
    test_repeated_source_sequence_ids_get_unique_document_identity()
    print("ByteScale HDP placement PASS")


if __name__ == "__main__":
    main()
