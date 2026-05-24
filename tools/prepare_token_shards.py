from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize text into uint32 .bin shards for PretrainingDataLoader.")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset id or local dataset path")
    parser.add_argument("--config", type=str, default=None, help="Optional HF dataset config")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--column", type=str, default="text")
    parser.add_argument("--tokenizer-name-or-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tokens-per-shard", type=int, default=100_000_000)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--eos-token", type=str, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokens_per_shard < 1:
        raise ValueError(f"tokens_per_shard must be >= 1, got {args.tokens_per_shard}")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {args.max_tokens}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)
    eos_id = tokenizer.convert_tokens_to_ids(args.eos_token) if args.eos_token is not None else tokenizer.eos_token_id

    dataset = load_dataset(args.dataset, name=args.config, split=args.split, streaming=args.streaming)
    if args.streaming:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)

    shard_idx = 0
    shard_tokens: list[int] = []
    total_tokens = 0
    for row in dataset:
        text = row.get(args.column)
        if not isinstance(text, str) or not text:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if eos_id is not None:
            token_ids.append(int(eos_id))
        for token_id in token_ids:
            shard_tokens.append(int(token_id))
            total_tokens += 1
            if len(shard_tokens) >= args.tokens_per_shard:
                _write_shard(output_dir, shard_idx, shard_tokens)
                shard_idx += 1
                shard_tokens = []
            if args.max_tokens is not None and total_tokens >= args.max_tokens:
                break
        if args.max_tokens is not None and total_tokens >= args.max_tokens:
            break

    if shard_tokens:
        _write_shard(output_dir, shard_idx, shard_tokens)
    print(f"wrote {total_tokens} tokens to {output_dir}")


def _write_shard(output_dir: Path, shard_idx: int, tokens: list[int]) -> None:
    path = output_dir / f"tokens_{shard_idx:05d}.bin"
    array = np.asarray(tokens, dtype=np.uint32)
    array.tofile(path)
    print(f"wrote {path} tokens={array.size}")


if __name__ == "__main__":
    main()
