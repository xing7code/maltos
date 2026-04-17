"""
To process HuggingFace Datasets:
    python3 tools/prepare_data.py --tokenizer-name-or-path Qwen/Qwen2.5-7B --output-folder datasets/emotion --n-tasks 16 hf --dataset dair-ai/emotion
To process Jsonl files:
    python3 tools/prepare_data.py --tokenizer-name-or-path Qwen/Qwen2.5-7B --output-folder datasets/c4-es --n-tasks 16 jsonl --dataset raw_datasets/c4-es-json-files
"""

import argparse

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.tokens import DocumentTokenizer


def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="Tokenizer")
    group.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.",
    )
    group.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="EOS token to add after each document. Default: None",
    )

    group = parser.add_argument_group(title="Output data")
    group.add_argument(
        "--output-folder", type=str, required=True, help="Path to the output folder to store the tokenized documents"
    )
    group = parser.add_argument_group(title="Miscellaneous configs")
    group.add_argument(
        "--logging-dir",
        type=str,
        default=None,
        help="Path to a folder for storing the logs of the preprocessing step. Default: None",
    )
    group.add_argument(
        "--n-tasks", type=int, default=8, help="Total number of tasks to run the preprocessing step. Default: 8"
    )
    group.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="Approximate target token budget for the final dataset (for example 500000000). Requires --estimated-total-tokens.",
    )
    group.add_argument(
        "--estimated-total-tokens",
        type=int,
        default=None,
        help="Estimated total tokens in the source dataset with your tokenizer. Used to derive sampling rate.",
    )
    group.add_argument(
        "--oversample-factor",
        type=float,
        default=1.05,
        help="Sampling safety margin when using --target-tokens (default: 1.05).",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by SamplerFilter when token budget sampling is enabled.",
    )
    # Subparsers for processing either Hugging Face datasets or jsonl files
    sp = parser.add_subparsers(
        dest="readers",
        required=True,
        description="Type of dataset to process. It can be either a Hugging Face Dataset loaded with datasets.load_data ('hf') or a .jsonl dataset ('jsonl')",
    )

    p1 = sp.add_parser(name="hf")
    p1.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to local stored dataset or repository on the Hugging Face hub that can be loaded with datasets.load_dataset",
    )
    p1.add_argument("--config", type=str, default=None, help="Optional HF dataset config/subset name")
    p1.add_argument("--column", type=str, default="text", help="Column to preprocess from the Dataset. Default: text")
    p1.add_argument("--split", type=str, default="train", help="Which split of the data to process. Default: train")

    p2 = sp.add_parser(name="jsonl")
    p2.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a .jsonl file or a folder containing multiple .jsonl files",
    )
    p2.add_argument("--column", type=str, default="text", help="Column to preprocess from the Dataset. Default: text")
    p2.add_argument(
        "--glob-pattern", type=str, default=None, help="A glob pattern to filter files to read. Default: None"
    )

    args = parser.parse_args()

    return args


def main(args):
    # Build datatrove reader
    if args.readers == "hf":
        datatrove_reader = HuggingFaceDatasetReader(
            dataset=args.dataset,
            text_key=args.column,
            dataset_options={"name": args.config, "split": args.split},
        )
    else:
        datatrove_reader = JsonlReader(data_folder=args.dataset, text_key=args.column, glob_pattern=args.glob_pattern)

    if (args.target_tokens is None) != (args.estimated_total_tokens is None):
        raise ValueError("--target-tokens and --estimated-total-tokens must be provided together.")

    pipeline = [datatrove_reader]

    if args.target_tokens is not None:
        if args.estimated_total_tokens <= 0:
            raise ValueError("--estimated-total-tokens must be > 0.")
        if args.oversample_factor <= 0:
            raise ValueError("--oversample-factor must be > 0.")

        sampling_rate = min(1.0, (args.target_tokens / args.estimated_total_tokens) * args.oversample_factor)
        print(
            f"[prepare_data] token-budget sampling enabled: target={args.target_tokens:,}, "
            f"estimated_total={args.estimated_total_tokens:,}, rate={sampling_rate:.6f}"
        )
        pipeline.append(SamplerFilter(rate=sampling_rate, seed=args.seed))

    pipeline.append(
        DocumentTokenizer(
            output_folder=args.output_folder,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            eos_token=args.eos_token,
            max_tokens_per_file=1e9,
        )
    )

    preprocess_executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=args.n_tasks,
        logging_dir=args.logging_dir,
    )
    preprocess_executor.run()


if __name__ == "__main__":
    _args = get_args()
    main(_args)
