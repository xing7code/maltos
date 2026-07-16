from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from data.sft import PackedSFTDataset, SFTDataLoader
from utils.constants import IGNORE_INDEX, INPUT_IDS_KEY, LABELS_KEY, POSITION_IDS_KEY, SEQUENCE_IDS_KEY, SFT_BATCH_KEYS
from utils.sft_messages import EncodedSFTExample
from utils.sft_packing import PackedSFTWriter, export_packing_metadata


def _write_meta(
    root: Path,
    *,
    summary,
    seq_len: int,
    sequences_per_shard: int,
) -> None:
    payload = {
        "format": "maltos_sft_packed",
        "format_version": 2,
        "source": {
            "dataset": {"path": "test", "config": None, "split": "train"},
            "tokenizer": {"name_or_path": "test", "vocab_size": 100, "eos_token_id": 99},
            "columns": {"messages": "messages", "prompt": "prompt", "completion": "completion"},
            "processing": {
                "assistant_only_loss": True,
                "apply_chat_template": True,
                "append_eos": True,
                "chat_template_file": None,
            },
            "stats": {"examples_seen": 0, "examples_encoded": 0, "examples_skipped": 0},
        },
        "packing": export_packing_metadata(
            summary=summary,
            seq_len=seq_len,
            sequences_per_shard=sequences_per_shard,
            packing_algorithm="next_fit",
            packing_buffer_size=1,
        ),
    }
    (root / "meta.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _populate_dataset_root(root: Path) -> None:
    writer = PackedSFTWriter(
        output_dir=root,
        seq_len=4,
        sequences_per_shard=2,
        max_sequences=None,
        pad_token_id=99,
        packing_algorithm="next_fit",
    )
    for token_ids in ([10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]):
        writer.add_example(
            EncodedSFTExample(
                token_ids=list(token_ids),
                supervised_mask=[0, 1, 1, 1],
            )
        )
    summary = writer.finish()
    _write_meta(root, summary=summary, seq_len=4, sequences_per_shard=2)


def test_sft_loader_reads_all_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _populate_dataset_root(root)
        dataset = PackedSFTDataset(root)
        loader = SFTDataLoader(dataset, micro_batch_size=2)
        batch = loader.next_batch()

        assert sorted(batch.keys()) == sorted(SFT_BATCH_KEYS)
        assert batch[INPUT_IDS_KEY].shape == (2, 4)
        assert batch[LABELS_KEY].shape == (2, 4)
        assert batch[POSITION_IDS_KEY].shape == (2, 4)
        assert batch[SEQUENCE_IDS_KEY].shape == (2, 4)
        assert batch[INPUT_IDS_KEY].tolist() == [[10, 11, 12, 13], [20, 21, 22, 23]]
        assert batch[LABELS_KEY].tolist() == [[11, 12, 13, IGNORE_INDEX], [21, 22, 23, IGNORE_INDEX]]
        assert batch[POSITION_IDS_KEY].tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]


def test_sft_loader_dp_partition_is_deterministic() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _populate_dataset_root(root)
        dataset = PackedSFTDataset(root)
        rank0 = SFTDataLoader(dataset, micro_batch_size=1, dp_rank=0, dp_world_size=2)
        rank1 = SFTDataLoader(dataset, micro_batch_size=1, dp_rank=1, dp_world_size=2)

        rank0_first = rank0.next_batch()[INPUT_IDS_KEY][0]
        rank1_first = rank1.next_batch()[INPUT_IDS_KEY][0]
        rank0_second = rank0.next_batch()[INPUT_IDS_KEY][0]
        rank1_second = rank1.next_batch()[INPUT_IDS_KEY][0]

        assert rank0_first.tolist() == [10, 11, 12, 13]
        assert rank1_first.tolist() == [20, 21, 22, 23]
        assert rank0_second.tolist() == [30, 31, 32, 33]
        assert rank1_second.tolist() == [40, 41, 42, 43]


def test_sft_loader_resume_restores_next_batch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _populate_dataset_root(root)
        dataset = PackedSFTDataset(root)
        continuous = SFTDataLoader(dataset, micro_batch_size=1)

        first_batch = continuous.next_batch()
        assert first_batch[INPUT_IDS_KEY][0].tolist() == [10, 11, 12, 13]
        saved_state = continuous.state_dict()
        continuous_second = continuous.next_batch()

        restored = SFTDataLoader(dataset, micro_batch_size=1, start_state=saved_state)
        restored_second = restored.next_batch()

        assert restored.state_dict() == continuous.state_dict()
        assert torch.equal(continuous_second[INPUT_IDS_KEY], restored_second[INPUT_IDS_KEY])
        assert torch.equal(continuous_second[LABELS_KEY], restored_second[LABELS_KEY])
        assert torch.equal(continuous_second[POSITION_IDS_KEY], restored_second[POSITION_IDS_KEY])
        assert torch.equal(continuous_second[SEQUENCE_IDS_KEY], restored_second[SEQUENCE_IDS_KEY])


def main() -> None:
    test_sft_loader_reads_all_fields()
    test_sft_loader_dp_partition_is_deterministic()
    test_sft_loader_resume_restores_next_batch()
    print("sft dataloader ok")


if __name__ == "__main__":
    main()
