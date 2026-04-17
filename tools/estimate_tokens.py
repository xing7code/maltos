import argparse
import math
from statistics import mean

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate total tokens for a HF dataset split.")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset repo id, e.g. HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config", type=str, default=None, help="Optional dataset config name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--column", type=str, default="text", help="Text column name")
    parser.add_argument("--tokenizer-name-or-path", type=str, required=True, help="Tokenizer path or HF repo id")
    parser.add_argument("--sample-size", type=int, default=2000, help="Number of sampled rows for estimation")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--streaming-buffer-size",
        type=int,
        default=10000,
        help="Shuffle buffer size for streaming mode (larger is more random but slower).",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=None,
        help="Total number of rows in the split. If set, estimated_total_tokens can be computed without metadata lookup.",
    )
    parser.add_argument("--eos-token", type=str, default=None, help="If set, count +1 token per document")
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="Optional. If provided, prints suggested sampling rate for prepare_data.py",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)

    ds = load_dataset(args.dataset, name=args.config, split=args.split, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.streaming_buffer_size)

    token_counts = []
    for row in ds:
        text = row.get(args.column)
        if not isinstance(text, str) or not text:
            continue
        n = len(tokenizer.encode(text, add_special_tokens=False))
        if args.eos_token is not None:
            n += 1
        token_counts.append(n)
        if len(token_counts) >= args.sample_size:
            break

    if not token_counts:
        raise ValueError("No valid text rows found in sample.")

    avg_tokens = mean(token_counts)

    print(f"dataset={args.dataset}")
    print(f"config={args.config}")
    print(f"split={args.split}")
    print(f"sample_size_used={len(token_counts)}")
    print(f"avg_tokens_per_doc={avg_tokens:.2f}")
    if args.num_rows is not None:
        est_total_tokens = math.ceil(avg_tokens * args.num_rows)
        print(f"num_rows={args.num_rows}")
        print(f"estimated_total_tokens={est_total_tokens}")
    else:
        est_total_tokens = None
        print("num_rows=unknown (pass --num-rows to estimate total tokens)")

    if args.target_tokens is not None:
        if est_total_tokens is None:
            print("suggested_sampling_rate=unknown (needs --num-rows)")
            print("suggested_prepare_data_args=unknown (needs --num-rows)")
        else:
            rate = min(1.0, args.target_tokens / est_total_tokens)
            print(f"suggested_sampling_rate={rate:.8f}")
            print(
                "suggested_prepare_data_args="
                f"--target-tokens {args.target_tokens} --estimated-total-tokens {est_total_tokens}"
            )


if __name__ == "__main__":
    main()
