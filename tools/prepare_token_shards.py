from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


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
    parser.add_argument("--expected-vocab-size", type=int, default=None)
    parser.add_argument("--eos-token", type=str, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--log-every-tokens", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokens_per_shard < 1:
        raise ValueError(f"tokens_per_shard must be >= 1, got {args.tokens_per_shard}")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {args.max_tokens}")
    if args.log_every_tokens < 1:
        raise ValueError(f"log_every_tokens must be >= 1, got {args.log_every_tokens}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)
    tokenizer_vocab_size = len(tokenizer)
    if args.expected_vocab_size is not None and tokenizer_vocab_size != args.expected_vocab_size:
        raise ValueError(
            "tokenizer vocab size mismatch: "
            f"tokenizer={args.tokenizer_name_or_path} vocab={tokenizer_vocab_size}, "
            f"expected={args.expected_vocab_size}"
        )
    eos_id = tokenizer.convert_tokens_to_ids(args.eos_token) if args.eos_token is not None else tokenizer.eos_token_id
    print(f"tokenizer={args.tokenizer_name_or_path} vocab_size={tokenizer_vocab_size} eos_id={eos_id}")

    dataset = load_dataset(args.dataset, name=args.config, split=args.split, streaming=args.streaming)
    if args.streaming:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)

    shard_idx = 0
    shard_tokens: list[int] = []
    total_tokens = 0
    total_docs = 0
    next_log_tokens = args.log_every_tokens
    start_time = time.perf_counter()
    for row in dataset:
        text = row.get(args.column)
        if not isinstance(text, str) or not text:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if eos_id is not None:
            token_ids.append(int(eos_id))
        total_docs += 1
        if args.max_tokens is not None:
            remaining = args.max_tokens - total_tokens
            token_ids = token_ids[:remaining]
        shard_tokens.extend(int(token_id) for token_id in token_ids)
        total_tokens += len(token_ids)
        while len(shard_tokens) >= args.tokens_per_shard:
            _write_shard(output_dir, shard_idx, shard_tokens[: args.tokens_per_shard])
            shard_idx += 1
            shard_tokens = shard_tokens[args.tokens_per_shard :]
        if total_tokens >= next_log_tokens:
            _log_progress(
                total_tokens=total_tokens,
                total_docs=total_docs,
                target_tokens=args.max_tokens,
                start_time=start_time,
            )
            while total_tokens >= next_log_tokens:
                next_log_tokens += args.log_every_tokens
        if args.max_tokens is not None and total_tokens >= args.max_tokens:
            break

    if shard_tokens:
        _write_shard(output_dir, shard_idx, shard_tokens)
    _log_progress(
        total_tokens=total_tokens,
        total_docs=total_docs,
        target_tokens=args.max_tokens,
        start_time=start_time,
        final=True,
    )
    print(f"wrote {total_tokens:,} tokens to {output_dir}")


def _write_shard(output_dir: Path, shard_idx: int, tokens: list[int]) -> None:
    path = output_dir / f"tokens_{shard_idx:05d}.bin"
    array = np.asarray(tokens, dtype=np.uint32)
    array.tofile(path)
    print(f"wrote {path} tokens={array.size}")


def _log_progress(
    *,
    total_tokens: int,
    total_docs: int,
    target_tokens: int | None,
    start_time: float,
    final: bool = False,
) -> None:
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    tokens_per_sec = total_tokens / elapsed
    prefix = "done" if final else "progress"
    message = (
        f"[{prefix}] tokens={total_tokens:,} docs={total_docs:,} "
        f"elapsed={_format_duration(elapsed)} tokens/sec={tokens_per_sec:,.0f}"
    )
    if target_tokens is not None and total_tokens > 0:
        remaining = max(target_tokens - total_tokens, 0)
        eta = remaining / max(tokens_per_sec, 1e-9)
        message += f" target={target_tokens:,} eta={_format_duration(eta)}"
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
