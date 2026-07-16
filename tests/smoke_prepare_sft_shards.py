from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from utils.sft_messages import (
    EncodedSFTExample,
    encode_messages_with_chat_template,
    encode_sft_example,
    extract_messages,
    has_supervised_assistant_turn,
    messages_from_prompt_completion,
)
from utils.constants import IGNORE_INDEX, INPUT_IDS_KEY, LABELS_KEY, PAD_SEQUENCE_ID, SEQUENCE_IDS_KEY
from utils.sft_packing import PackedSFTWriter, export_packing_metadata


class FakeChatTokenizer:
    def __init__(self) -> None:
        self.chat_template = "{{ messages }}"
        self.eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        assert len(messages) == 2
        return {
            "input_ids": [11, 12, 21, 22, 23],
            "assistant_masks": [0, 0, 1, 1, 1],
        }

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [ord(ch) for ch in text]


class FakeBrokenChatTokenizer:
    def __init__(self) -> None:
        self.chat_template = "{{ messages }}"
        self.eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return {
            "input_ids": [11, 12, 21, 22, 23],
            "assistant_masks": [0, 0, 0, 0, 0],
        }


def _read_shard_field(root: Path, shard, field_name: str) -> np.memmap:
    field = getattr(shard, field_name)
    return np.memmap(
        root / shard.path,
        mode="r",
        dtype=np.dtype(field.dtype),
        shape=field.shape,
        offset=field.offset,
    )


def test_extract_messages_accepts_json_string() -> None:
    messages = extract_messages(
        {
            "messages": '[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"text","text":"ok"}]}]'
        },
        "messages",
    )
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]


def test_chat_template_marks_only_assistant_tokens() -> None:
    tokenizer = FakeChatTokenizer()
    encoded = encode_messages_with_chat_template(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        tokenizer=tokenizer,
        assistant_only_loss=True,
    )
    assert encoded.token_ids == [11, 12, 21, 22, 23]
    assert encoded.supervised_mask == [0, 0, 1, 1, 1]


def test_chat_template_rejects_all_zero_assistant_mask() -> None:
    tokenizer = FakeBrokenChatTokenizer()
    try:
        encode_messages_with_chat_template(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            tokenizer=tokenizer,
            assistant_only_loss=True,
        )
    except ValueError as exc:
        assert "all values are 0" in str(exc)
        return
    raise AssertionError("expected all-zero assistant mask to raise")


def test_prompt_completion_converts_to_messages() -> None:
    messages = messages_from_prompt_completion("A", "B")
    assert messages == [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
    ]


def test_has_supervised_assistant_turn_requires_non_empty_assistant() -> None:
    assert has_supervised_assistant_turn(
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
        ]
    )
    assert not has_supervised_assistant_turn(
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": ""},
        ]
    )
    assert not has_supervised_assistant_turn(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "A"},
        ]
    )


def test_encode_sft_example_skips_rows_without_non_empty_assistant_turn() -> None:
    tokenizer = FakeBrokenChatTokenizer()
    encoded = encode_sft_example(
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": ""},
            ]
        },
        tokenizer=tokenizer,
        messages_column="messages",
        prompt_column="prompt",
        completion_column="completion",
        apply_chat_template=True,
        assistant_only_loss=True,
        append_eos=True,
    )
    assert encoded is None


def test_writer_pads_row_before_crossing_example_boundary() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        writer = PackedSFTWriter(
            output_dir=Path(tmp),
            seq_len=6,
            sequences_per_shard=2,
            max_sequences=None,
            pad_token_id=99,
            packing_algorithm="next_fit",
        )
        writer.add_example(
            EncodedSFTExample(
                token_ids=[10, 11, 12],
                supervised_mask=[0, 1, 1],
            )
        )
        writer.add_example(
            EncodedSFTExample(
                token_ids=[20, 21, 22, 23],
                supervised_mask=[0, 1, 1, 1],
            )
        )
        summary = writer.finish()

        assert summary.packed_sequences == 2
        assert summary.padded_tokens == 5
        assert summary.dropped_tail_tokens == 0
        shard = summary.shards[0]
        input_ids = _read_shard_field(Path(tmp), shard, "input_field")
        labels = _read_shard_field(Path(tmp), shard, "label_field")
        position_ids = _read_shard_field(Path(tmp), shard, "position_field")
        sequence_ids = _read_shard_field(Path(tmp), shard, "sequence_field")
        assert input_ids.shape == (2, 6)
        assert input_ids.tolist() == [[10, 11, 12, 99, 99, 99], [20, 21, 22, 23, 99, 99]]
        assert labels.tolist() == [
            [11, 12, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
            [21, 22, 23, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
        ]
        assert position_ids.tolist() == [[0, 1, 2, 0, 0, 0], [0, 1, 2, 3, 0, 0]]
        assert sequence_ids.tolist() == [
            [0, 0, 0, PAD_SEQUENCE_ID, PAD_SEQUENCE_ID, PAD_SEQUENCE_ID],
            [0, 0, 0, 0, PAD_SEQUENCE_ID, PAD_SEQUENCE_ID],
        ]


def test_writer_keeps_multiple_examples_in_same_row_when_they_fit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        writer = PackedSFTWriter(
            output_dir=Path(tmp),
            seq_len=8,
            sequences_per_shard=2,
            max_sequences=None,
            pad_token_id=99,
            packing_algorithm="next_fit",
        )
        writer.add_example(
            EncodedSFTExample(
                token_ids=[10, 11, 12],
                supervised_mask=[0, 1, 1],
            )
        )
        writer.add_example(
            EncodedSFTExample(
                token_ids=[20, 21, 22],
                supervised_mask=[0, 1, 1],
            )
        )
        summary = writer.finish()

        assert summary.packed_sequences == 1
        assert summary.padded_tokens == 2
        shard = summary.shards[0]
        input_ids = _read_shard_field(Path(tmp), shard, "input_field")
        labels = _read_shard_field(Path(tmp), shard, "label_field")
        position_ids = _read_shard_field(Path(tmp), shard, "position_field")
        sequence_ids = _read_shard_field(Path(tmp), shard, "sequence_field")
        assert input_ids.shape == (1, 8)
        assert input_ids.tolist() == [[10, 11, 12, 20, 21, 22, 99, 99]]
        assert labels.tolist() == [[11, 12, IGNORE_INDEX, 21, 22, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]]
        assert position_ids.tolist() == [[0, 1, 2, 0, 1, 2, 0, 0]]
        assert sequence_ids.tolist() == [[0, 0, 0, 1, 1, 1, PAD_SEQUENCE_ID, PAD_SEQUENCE_ID]]


def test_writer_best_fit_decreasing_reduces_padding_vs_next_fit() -> None:
    examples = [
        EncodedSFTExample(token_ids=[10, 11, 12, 13, 14, 15], supervised_mask=[0, 1, 1, 1, 1, 1]),
        EncodedSFTExample(token_ids=[20, 21, 22, 23, 24, 25], supervised_mask=[0, 1, 1, 1, 1, 1]),
        EncodedSFTExample(token_ids=[30, 31, 32, 33], supervised_mask=[0, 1, 1, 1]),
        EncodedSFTExample(token_ids=[40, 41, 42, 43], supervised_mask=[0, 1, 1, 1]),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        next_fit = PackedSFTWriter(
            output_dir=Path(tmp),
            seq_len=10,
            sequences_per_shard=8,
            max_sequences=None,
            pad_token_id=99,
            packing_algorithm="next_fit",
        )
        for example in examples:
            next_fit.add_example(example)
        next_fit_summary = next_fit.finish()

    with tempfile.TemporaryDirectory() as tmp:
        best_fit = PackedSFTWriter(
            output_dir=Path(tmp),
            seq_len=10,
            sequences_per_shard=8,
            max_sequences=None,
            pad_token_id=99,
            packing_algorithm="best_fit_decreasing",
            packing_buffer_size=16,
        )
        for example in examples:
            best_fit.add_example(example)
        best_fit_summary = best_fit.finish()
        shard = best_fit_summary.shards[0]
        input_ids = _read_shard_field(Path(tmp), shard, "input_field")

        assert next_fit_summary.packed_sequences == 3
        assert next_fit_summary.padded_tokens == 10
        assert best_fit_summary.packed_sequences == 2
        assert best_fit_summary.padded_tokens == 0
        assert input_ids.tolist() == [
            [10, 11, 12, 13, 14, 15, 30, 31, 32, 33],
            [20, 21, 22, 23, 24, 25, 40, 41, 42, 43],
        ]


def test_export_packing_metadata_is_structured() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        writer = PackedSFTWriter(
            output_dir=Path(tmp),
            seq_len=8,
            sequences_per_shard=2,
            max_sequences=None,
            pad_token_id=99,
            packing_algorithm="best_fit_decreasing",
            packing_buffer_size=16,
        )
        writer.add_example(
            EncodedSFTExample(
                token_ids=[10, 11, 12],
                supervised_mask=[0, 1, 1],
            )
        )
        writer.add_example(
            EncodedSFTExample(
                token_ids=[20, 21, 22],
                supervised_mask=[0, 1, 1],
            )
        )
        summary = writer.finish()
        metadata = export_packing_metadata(
            summary=summary,
            seq_len=8,
            sequences_per_shard=2,
            packing_algorithm="best_fit_decreasing",
            packing_buffer_size=16,
        )

        assert metadata["layout"]["seq_len"] == 8
        assert metadata["layout"]["fields"][LABELS_KEY]["ignore_index"] == IGNORE_INDEX
        assert metadata["layout"]["fields"][SEQUENCE_IDS_KEY]["pad_sequence_id"] == PAD_SEQUENCE_ID
        assert metadata["strategy"]["algorithm"] == "best_fit_decreasing"
        assert metadata["strategy"]["name"] == "example_aware_windowed_best_fit_decreasing"
        assert metadata["stats"]["packed_sequences"] == 1
        assert metadata["stats"]["num_shards"] == 1
        assert metadata["shards"][0]["fields"][INPUT_IDS_KEY]["shape"] == [1, 8]


def main() -> None:
    test_extract_messages_accepts_json_string()
    test_chat_template_marks_only_assistant_tokens()
    test_chat_template_rejects_all_zero_assistant_mask()
    test_prompt_completion_converts_to_messages()
    test_has_supervised_assistant_turn_requires_non_empty_assistant()
    test_encode_sft_example_skips_rows_without_non_empty_assistant_turn()
    test_writer_pads_row_before_crossing_example_boundary()
    test_writer_keeps_multiple_examples_in_same_row_when_they_fit()
    test_writer_best_fit_decreasing_reduces_padding_vs_next_fit()
    test_export_packing_metadata_is_structured()
    print("prepare_sft_shards smoke ok")


if __name__ == "__main__":
    main()
