from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from utils.sft_messages import encode_sft_example
from utils.sft_packing import (
    PACKING_ALGORITHMS,
    PackedSFTSummary,
    PackedSFTWriter,
    export_packing_metadata,
    resolve_pad_token_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare fixed-length SFT input/label shards for future MALTOS SFT dataloaders."
    )
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset id, local dataset path, or json/jsonl/parquet/csv file")
    parser.add_argument("--config", type=str, default=None, help="Optional HF dataset config")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--messages-column", type=str, default="messages")
    parser.add_argument("--prompt-column", type=str, default="prompt")
    parser.add_argument("--completion-column", type=str, default="completion")
    parser.add_argument("--tokenizer-name-or-path", type=str, required=True)
    parser.add_argument(
        "--chat-template-file",
        type=str,
        default=None,
        help="Optional local Jinja chat template file to override tokenizer.chat_template",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seq-len", type=int, required=True, help="Packed training sequence length")
    parser.add_argument("--sequences-per-shard", type=int, default=8192)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--expected-vocab-size", type=int, default=None)
    parser.add_argument(
        "--packing-algorithm",
        type=str,
        default="best_fit_decreasing",
        choices=sorted(PACKING_ALGORITHMS),
        help="Sequence packing algorithm. best_fit_decreasing is a bounded-window offline bin packer.",
    )
    parser.add_argument(
        "--packing-buffer-size",
        type=int,
        default=4096,
        help="Number of segmented examples to buffer before running best_fit_decreasing.",
    )
    parser.add_argument("--assistant-only-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render message/prompt-completion rows through tokenizer chat template when available",
    )
    parser.add_argument(
        "--append-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append eos when using plain prompt/completion fallback",
    )
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--log-every-examples", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {args.seq_len}")
    if args.sequences_per_shard < 1:
        raise ValueError(f"sequences_per_shard must be >= 1, got {args.sequences_per_shard}")
    if args.max_examples is not None and args.max_examples < 1:
        raise ValueError(f"max_examples must be >= 1, got {args.max_examples}")
    if args.max_sequences is not None and args.max_sequences < 1:
        raise ValueError(f"max_sequences must be >= 1, got {args.max_sequences}")
    if args.packing_buffer_size < 1:
        raise ValueError(f"packing_buffer_size must be >= 1, got {args.packing_buffer_size}")
    if args.log_every_examples < 1:
        raise ValueError(f"log_every_examples must be >= 1, got {args.log_every_examples}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)
    if args.chat_template_file is not None:
        tokenizer.chat_template = Path(args.chat_template_file).read_text(encoding="utf-8")
    tokenizer_vocab_size = len(tokenizer)
    if args.expected_vocab_size is not None and tokenizer_vocab_size != args.expected_vocab_size:
        raise ValueError(
            "tokenizer vocab size mismatch: "
            f"tokenizer={args.tokenizer_name_or_path} vocab={tokenizer_vocab_size}, "
            f"expected={args.expected_vocab_size}"
        )
    print(
        f"tokenizer={args.tokenizer_name_or_path} "
        f"vocab_size={tokenizer_vocab_size} eos_id={tokenizer.eos_token_id} "
        f"pad_id={resolve_pad_token_id(tokenizer)} "
        f"chat_template={'yes' if getattr(tokenizer, 'chat_template', None) else 'no'} "
        f"packing={args.packing_algorithm} buffer={args.packing_buffer_size}"
    )

    dataset = _load_dataset(args)
    if args.streaming:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)

    packer = PackedSFTWriter(
        output_dir=output_dir,
        seq_len=args.seq_len,
        sequences_per_shard=args.sequences_per_shard,
        max_sequences=args.max_sequences,
        pad_token_id=resolve_pad_token_id(tokenizer),
        packing_algorithm=args.packing_algorithm,
        packing_buffer_size=args.packing_buffer_size,
    )

    examples_seen = 0
    examples_encoded = 0
    examples_skipped = 0
    start_time = time.perf_counter()
    next_log_examples = args.log_every_examples
    for row in dataset:
        if args.max_examples is not None and examples_seen >= args.max_examples:
            break
        examples_seen += 1
        encoded = encode_sft_example(
            row,
            tokenizer=tokenizer,
            messages_column=args.messages_column,
            prompt_column=args.prompt_column,
            completion_column=args.completion_column,
            apply_chat_template=args.apply_chat_template,
            assistant_only_loss=args.assistant_only_loss,
            append_eos=args.append_eos,
        )
        if encoded is None:
            examples_skipped += 1
        else:
            examples_encoded += 1
            packer.add_example(encoded)
        if examples_seen >= next_log_examples:
            _log_progress(
                examples_seen=examples_seen,
                examples_encoded=examples_encoded,
                examples_skipped=examples_skipped,
                packer=packer,
                start_time=start_time,
            )
            while examples_seen >= next_log_examples:
                next_log_examples += args.log_every_examples
        if packer.reached_sequence_limit:
            break

    summary = packer.finish()
    _write_meta(
        output_dir / "meta.json",
        args=args,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        tokenizer_vocab_size=tokenizer_vocab_size,
        eos_token_id=tokenizer.eos_token_id,
        examples_seen=examples_seen,
        examples_encoded=examples_encoded,
        examples_skipped=examples_skipped,
        summary=summary,
    )
    _log_progress(
        examples_seen=examples_seen,
        examples_encoded=examples_encoded,
        examples_skipped=examples_skipped,
        packer=packer,
        start_time=start_time,
        final=True,
    )
    print(
        f"wrote {summary.packed_sequences:,} sequences "
        f"({summary.packed_input_tokens:,} input tokens, "
        f"{summary.packed_supervised_tokens:,} supervised targets) to {output_dir}"
    )

def _load_dataset(args: argparse.Namespace):
    from datasets import DatasetDict, load_dataset, load_from_disk

    path = Path(args.dataset)
    if path.exists():
        if path.is_file():
            loader_name = _dataset_loader_for_suffix(path.suffix)
            return load_dataset(loader_name, data_files=str(path), split=args.split, streaming=args.streaming)
        if args.streaming:
            raise ValueError("streaming is not supported for local load_from_disk datasets")
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            if args.split not in dataset:
                raise ValueError(f"split={args.split!r} not found in local dataset; available={sorted(dataset.keys())}")
            return dataset[args.split]
        if args.split != "train":
            raise ValueError("split selection for local on-disk Dataset is only supported when it is a DatasetDict")
        return dataset
    return load_dataset(args.dataset, name=args.config, split=args.split, streaming=args.streaming)


def _dataset_loader_for_suffix(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"unsupported local dataset file suffix={suffix!r}")


def _write_meta(
    path: Path,
    *,
    args: argparse.Namespace,
    tokenizer_name_or_path: str,
    tokenizer_vocab_size: int,
    eos_token_id: int | None,
    examples_seen: int,
    examples_encoded: int,
    examples_skipped: int,
    summary: PackedSFTSummary,
) -> None:
    payload = {
        "format": "maltos_sft_packed",
        "format_version": 2,
        "source": {
            "dataset": {
                "path": args.dataset,
                "config": args.config,
                "split": args.split,
            },
            "tokenizer": {
                "name_or_path": tokenizer_name_or_path,
                "vocab_size": tokenizer_vocab_size,
                "eos_token_id": eos_token_id,
            },
            "columns": {
                "messages": args.messages_column,
                "prompt": args.prompt_column,
                "completion": args.completion_column,
            },
            "processing": {
                "assistant_only_loss": bool(args.assistant_only_loss),
                "apply_chat_template": bool(args.apply_chat_template),
                "append_eos": bool(args.append_eos),
                "chat_template_file": args.chat_template_file,
            },
            "stats": {
                "examples_seen": examples_seen,
                "examples_encoded": examples_encoded,
                "examples_skipped": examples_skipped,
            },
        },
        "packing": export_packing_metadata(
            summary=summary,
            seq_len=args.seq_len,
            sequences_per_shard=args.sequences_per_shard,
            packing_algorithm=args.packing_algorithm,
            packing_buffer_size=args.packing_buffer_size,
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _log_progress(
    *,
    examples_seen: int,
    examples_encoded: int,
    examples_skipped: int,
    packer: PackedSFTWriter,
    start_time: float,
    final: bool = False,
) -> None:
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    prefix = "done" if final else "progress"
    examples_per_sec = examples_seen / elapsed
    tokens_per_sec = packer.raw_tokens / elapsed
    message = (
        f"[{prefix}] examples={examples_seen:,} encoded={examples_encoded:,} skipped={examples_skipped:,} "
        f"raw_tokens={packer.raw_tokens:,} raw_supervised_tokens={packer.raw_supervised_tokens:,} "
        f"packed_sequences={packer.packed_sequences:,} packed_tokens={packer.packed_sequences * packer.seq_len:,} "
        f"packed_supervised_tokens={packer.packed_supervised_tokens:,} padded_tokens={packer.padded_tokens:,} "
        f"elapsed={_format_duration(elapsed)} "
        f"examples/sec={examples_per_sec:,.1f} tokens/sec={tokens_per_sec:,.0f}"
    )
    print(message)


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()
