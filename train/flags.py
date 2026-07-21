from __future__ import annotations

import argparse
from typing import Any

import yaml

from parallel.context_interfaces import ContextParallelAttentionCoreType
from utils.attention_backend import ATTENTION_BACKEND_CHOICES, AttentionBackend

TRAIN_CLI_RUNTIME_SPEC_FORMAT = "train_cli_args"
TRAIN_CLI_RUNTIME_SPEC_VERSION = 1

_RUNTIME_SPEC_MODEL_FIELDS = (
    "model",
    "dim",
    "n_heads",
    "n_kv_heads",
    "hidden_size",
    "n_layers",
    "num_experts",
    "moe_aux_loss_coef",
    "vocab_size",
    "eps",
    "seq_len",
    "attention_backend",
    "activation_checkpointing",
    "activation_checkpoint_every_n_layers",
    "use_sp",
    "tp_size",
)

_RUNTIME_SPEC_RUNTIME_FIELDS = (
    "dp_size",
    "tp_size",
    "pp_size",
    "cp_size",
    "ep_size",
    "pp_microbatches",
    "cp_attn_core",
    "zero_stage",
    "ddp_mode",
    "precision",
    "use_sp",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM training recipe.")
    parser.add_argument("--config", type=str, default=None, help="YAML training recipe")
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        default=None,
        help="Token shard .bin files or directories, or a packed SFT dataset directory/meta.json",
    )
    parser.add_argument("--data-format", type=str, default="auto", choices=("auto", "pretrain", "sft"))
    parser.add_argument("--token-dtype", type=str, default="uint32", choices=("uint16", "uint32", "int64"))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr-schedule", type=str, default="constant", choices=("constant", "linear", "cosine"))
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model", type=str, default="tiny", choices=("tiny", "tiny_moe", "llama", "olmo", "olmo2"))
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--moe-aux-loss-coef", type=float, default=0.0)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=AttentionBackend.SDPA_AUTO,
        choices=ATTENTION_BACKEND_CHOICES,
    )
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--activation-checkpoint-every-n-layers", type=int, default=1)

    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--pp-microbatches", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument(
        "--cp-attn-core",
        type=str,
        default="all_gather_kv",
        choices=tuple(core.value for core in ContextParallelAttentionCoreType),
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--zero-stage", type=int, default=0, choices=(0, 1, 2, 3))
    parser.add_argument("--use-sp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ddp-mode", type=str, default=None, choices=("sync", "async", "bucket"))
    parser.add_argument("--precision", type=str, default="fp32", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--disable-metrics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-profiler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-profiler-dir", type=str, default="traces/train")
    parser.add_argument("--torch-profiler-wait", type=int, default=1)
    parser.add_argument("--torch-profiler-warmup", type=int, default=1)
    parser.add_argument("--torch-profiler-active", type=int, default=3)
    parser.add_argument("--torch-profiler-repeat", type=int, default=1)
    parser.add_argument("--torch-profiler-record-shapes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-profiler-profile-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-profiler-with-stack", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-profiler-with-flops", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-profiler-rank0-only", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--metrics-jsonl", type=str, default=None)
    parser.add_argument("--run-manifest", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None, choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated W&B tags")
    parser.add_argument("--wandb-checkpoint-every", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--checkpoint-keep-last", type=int, default=None)
    parser.add_argument("--checkpoint-keep-every-n-steps", type=int, default=None)
    parser.add_argument("--checkpoint-min-free-gb", type=float, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--load-weights-only", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="29550")
    return parser


def parse_args_with_defaults(
    defaults: dict[str, Any],
    argv: list[str] | None = None,
    *,
    require_data: bool = True,
) -> argparse.Namespace:
    parser = build_arg_parser()
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    _validate_data_requirement(args, require_data=require_data)
    return args


def parse_args_from(argv: list[str] | None = None, *, require_data: bool = True) -> argparse.Namespace:
    config_path = _preparse_config_path(argv)
    defaults = _load_config_defaults(config_path) if config_path is not None else {}
    return parse_args_with_defaults(defaults, argv, require_data=require_data)


def parse_args(*, require_data: bool = True) -> argparse.Namespace:
    return parse_args_from(require_data=require_data)


def build_runtime_spec(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "format": TRAIN_CLI_RUNTIME_SPEC_FORMAT,
        "version": TRAIN_CLI_RUNTIME_SPEC_VERSION,
        "model": {name: getattr(args, name) for name in _RUNTIME_SPEC_MODEL_FIELDS},
        "runtime": {name: getattr(args, name) for name in _RUNTIME_SPEC_RUNTIME_FIELDS},
    }


def parse_runtime_spec_args(
    spec: dict[str, Any],
) -> argparse.Namespace:
    if spec.get("format") != TRAIN_CLI_RUNTIME_SPEC_FORMAT:
        raise ValueError(f"unsupported runtime spec format={spec.get('format')!r}")
    version = int(spec.get("version", 0))
    if version != TRAIN_CLI_RUNTIME_SPEC_VERSION:
        raise ValueError(f"unsupported runtime spec version={version}")
    model_defaults = spec.get("model")
    runtime_defaults = spec.get("runtime")
    if not isinstance(model_defaults, dict):
        raise ValueError("runtime spec missing model mapping")
    if not isinstance(runtime_defaults, dict):
        raise ValueError("runtime spec missing runtime mapping")
    _validate_runtime_spec_fields("model", model_defaults, _RUNTIME_SPEC_MODEL_FIELDS)
    _validate_runtime_spec_fields("runtime", runtime_defaults, _RUNTIME_SPEC_RUNTIME_FIELDS)
    for name in set(model_defaults) & set(runtime_defaults):
        if model_defaults[name] != runtime_defaults[name]:
            raise ValueError(
                f"runtime spec field {name!r} disagrees between model and runtime sections"
            )
    return argparse.Namespace(**{**model_defaults, **runtime_defaults})


def _validate_runtime_spec_fields(section: str, values: dict[str, Any], expected_fields: tuple[str, ...]) -> None:
    expected = set(expected_fields)
    actual = set(values)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"runtime spec {section} fields mismatch: missing={missing}, extra={extra}")


def _preparse_config_path(argv: list[str] | None = None) -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args(argv)
    return args.config


def _load_config_defaults(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a YAML mapping, got {type(raw)!r}")
    defaults: dict[str, Any] = {"config": path}
    for section, values in raw.items():
        if not isinstance(values, dict):
            defaults[section] = values
            continue
        for key, value in values.items():
            defaults[_config_key_to_arg_dest(section, key)] = value
    return defaults


def _validate_data_requirement(args: argparse.Namespace, *, require_data: bool) -> None:
    if require_data and args.data is None:
        raise ValueError("--data is required unless provided by --config")


def _config_key_to_arg_dest(section: str, key: str) -> str:
    aliases = {
        ("data", "paths"): "data",
        ("data", "format"): "data_format",
        ("data", "seq_len"): "seq_len",
        ("data", "micro_batch_size"): "micro_batch_size",
        ("model", "type"): "model",
        ("model", "hidden_size"): "dim",
        ("model", "dim"): "dim",
        ("model", "intermediate_size"): "hidden_size",
        ("model", "num_layers"): "n_layers",
        ("model", "n_layers"): "n_layers",
        ("model", "num_experts"): "num_experts",
        ("model", "moe_aux_loss_coef"): "moe_aux_loss_coef",
        ("model", "num_heads"): "n_heads",
        ("model", "n_heads"): "n_heads",
        ("model", "num_kv_heads"): "n_kv_heads",
        ("model", "n_kv_heads"): "n_kv_heads",
        ("model", "vocab_size"): "vocab_size",
        ("model", "rms_norm_eps"): "eps",
        ("model", "eps"): "eps",
        ("model", "attention_backend"): "attention_backend",
        ("model", "activation_checkpointing"): "activation_checkpointing",
        ("model", "activation_checkpoint_every_n_layers"): "activation_checkpoint_every_n_layers",
        ("parallel", "dp_size"): "dp_size",
        ("parallel", "pp_size"): "pp_size",
        ("parallel", "pp_microbatches"): "pp_microbatches",
        ("parallel", "cp_size"): "cp_size",
        ("parallel", "cp_attn_core"): "cp_attn_core",
        ("parallel", "tp_size"): "tp_size",
        ("parallel", "ep_size"): "ep_size",
        ("parallel", "use_sp"): "use_sp",
        ("parallel", "zero_stage"): "zero_stage",
        ("parallel", "ddp_mode"): "ddp_mode",
        ("training", "max_steps"): "max_steps",
        ("training", "dry_run"): "dry_run",
        ("training", "grad_accum_steps"): "grad_accum_steps",
        ("training", "lr"): "lr",
        ("training", "weight_decay"): "weight_decay",
        ("training", "adam_beta1"): "adam_beta1",
        ("training", "adam_beta2"): "adam_beta2",
        ("training", "adam_eps"): "adam_eps",
        ("training", "fused_adamw"): "fused_adamw",
        ("training", "lr_schedule"): "lr_schedule",
        ("training", "warmup_steps"): "warmup_steps",
        ("training", "min_lr"): "min_lr",
        ("training", "precision"): "precision",
        ("training", "grad_clip"): "grad_clip",
        ("training", "disable_metrics"): "disable_metrics",
        ("profiling", "torch_profiler"): "torch_profiler",
        ("profiling", "torch_profiler_dir"): "torch_profiler_dir",
        ("profiling", "torch_profiler_wait"): "torch_profiler_wait",
        ("profiling", "torch_profiler_warmup"): "torch_profiler_warmup",
        ("profiling", "torch_profiler_active"): "torch_profiler_active",
        ("profiling", "torch_profiler_repeat"): "torch_profiler_repeat",
        ("profiling", "torch_profiler_record_shapes"): "torch_profiler_record_shapes",
        ("profiling", "torch_profiler_profile_memory"): "torch_profiler_profile_memory",
        ("profiling", "torch_profiler_with_stack"): "torch_profiler_with_stack",
        ("profiling", "torch_profiler_with_flops"): "torch_profiler_with_flops",
        ("profiling", "torch_profiler_rank0_only"): "torch_profiler_rank0_only",
        ("logging", "log_every"): "log_every",
        ("logging", "metrics_jsonl"): "metrics_jsonl",
        ("logging", "run_manifest"): "run_manifest",
        ("logging", "wandb_project"): "wandb_project",
        ("logging", "wandb_run_name"): "wandb_run_name",
        ("logging", "wandb_entity"): "wandb_entity",
        ("logging", "wandb_run_id"): "wandb_run_id",
        ("logging", "wandb_mode"): "wandb_mode",
        ("logging", "wandb_tags"): "wandb_tags",
        ("logging", "wandb_checkpoint_every"): "wandb_checkpoint_every",
        ("checkpoint", "dir"): "checkpoint_dir",
        ("checkpoint", "checkpoint_dir"): "checkpoint_dir",
        ("checkpoint", "every"): "checkpoint_every",
        ("checkpoint", "checkpoint_every"): "checkpoint_every",
        ("checkpoint", "keep_last"): "checkpoint_keep_last",
        ("checkpoint", "checkpoint_keep_last"): "checkpoint_keep_last",
        ("checkpoint", "keep_every_n_steps"): "checkpoint_keep_every_n_steps",
        ("checkpoint", "checkpoint_keep_every_n_steps"): "checkpoint_keep_every_n_steps",
        ("checkpoint", "min_free_gb"): "checkpoint_min_free_gb",
        ("checkpoint", "checkpoint_min_free_gb"): "checkpoint_min_free_gb",
        ("checkpoint", "resume_from"): "resume_from",
        ("checkpoint", "load_weights_only"): "load_weights_only",
    }
    return aliases.get((section, key), key)
