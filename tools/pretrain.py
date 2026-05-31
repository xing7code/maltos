from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml

from data import PretrainingDataLoader, TokenShardDataset
from models import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaForCausalLMTp,
    LlamaForCausalLMTpSp,
    TinyTransformer,
    TinyTransformerTp,
    TinyTransformerTpSp,
)
from parallel import ParallelPlan
from runtime import MeshConfig, RuntimeCore
from runtime.layers.tp import ColumnParallelLinear, RowParallelLinear
from runtime.plugins.ddp import BucketDataParallelPlugin, DataParallelPlugin
from runtime.plugins.grad_clip import GradClipPlugin
from runtime.plugins.perf_metrics import PerfMetricsPlugin
from runtime.plugins.precision import PrecisionPlugin
from runtime.plugins.sp import SequenceParallelPlugin
from runtime.plugins.tp import TensorParallelPlugin
from runtime.plugins.zero1 import Zero1Plugin
from runtime.plugins.zero2 import Zero2Plugin
from runtime.plugins.zero3 import Zero3Plugin
from train import Trainer, TrainerConfig
from utils.metrics import (
    ConsoleMetricLogger,
    JsonlMetricLogger,
    MetricLogger,
    WandbCheckpointUploader,
    WandbMetricLogger,
)
from utils.distributed import distributed_barrier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM pretraining recipe.")
    parser.add_argument("--config", type=str, default=None, help="YAML training recipe")
    parser.add_argument("--data", type=str, nargs="+", default=None, help="Token shard .bin files or directories")
    parser.add_argument("--token-dtype", type=str, default="uint32", choices=("uint16", "uint32", "int64"))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model", type=str, default="tiny", choices=("tiny", "llama"))
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--eps", type=float, default=1e-5)

    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--zero-stage", type=int, default=0, choices=(0, 1, 2, 3))
    parser.add_argument("--use-sp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ddp-mode", type=str, default=None, choices=("sync", "async", "bucket"))
    parser.add_argument("--precision", type=str, default="fp32", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--disable-perf-metrics", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--metrics-jsonl", type=str, default=None)
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

    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="29550")
    config_path = _preparse_config_path()
    if config_path is not None:
        parser.set_defaults(**_load_config_defaults(config_path))
    args = parser.parse_args()
    if args.data is None:
        raise ValueError("--data is required unless provided by --config")
    return args


def main() -> None:
    args = parse_args()
    _maybe_init_distributed(args)
    device = _select_device()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size != args.dp_size * args.tp_size:
        raise ValueError(f"world_size={world_size} must equal dp_size * tp_size={args.dp_size * args.tp_size}")
    if args.use_sp and args.tp_size <= 1:
        raise ValueError("--use-sp requires --tp-size > 1")
    if args.zero_stage > 0 and args.dp_size <= 1:
        raise ValueError("--zero-stage > 0 requires --dp-size > 1")
    if args.zero_stage > 0 and args.ddp_mode is not None:
        raise ValueError("--ddp-mode is only valid when --zero-stage is 0")
    if args.wandb_checkpoint_every is not None:
        if args.wandb_checkpoint_every < 1:
            raise ValueError("--wandb-checkpoint-every must be >= 1")
        if args.checkpoint_every is None:
            raise ValueError("--wandb-checkpoint-every requires --checkpoint-every")
        if args.wandb_checkpoint_every % args.checkpoint_every != 0:
            raise ValueError("--wandb-checkpoint-every must be a multiple of --checkpoint-every")
        if args.wandb_project is None or args.wandb_mode == "disabled":
            raise ValueError("--wandb-checkpoint-every requires enabled W&B logging")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    dp_rank = rank // args.tp_size
    data_paths = _expand_data_paths(args.data)
    model = _build_model(args)
    initial_trainable_params = _count_trainable_params(model)
    loader = PretrainingDataLoader(
        TokenShardDataset(data_paths, dtype=np.dtype(args.token_dtype)),
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        dp_rank=dp_rank,
        dp_world_size=args.dp_size,
        seed=args.seed,
    )
    runtime = _build_runtime(args, model, device)
    logger, checkpoint_uploader = _build_logging(args, rank)
    trainer = Trainer(
        runtime=runtime,
        dataloader=loader,
        config=TrainerConfig(
            max_steps=args.max_steps,
            log_every=args.log_every,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume_from,
            checkpoint_keep_last=args.checkpoint_keep_last,
            checkpoint_keep_every_n_steps=args.checkpoint_keep_every_n_steps,
            checkpoint_min_free_gb=args.checkpoint_min_free_gb,
        ),
        logger=logger,
        checkpoint_uploader=checkpoint_uploader,
    )
    trainer.setup()
    _print_run_summary(
        args=args,
        runtime=runtime,
        initial_trainable_params=initial_trainable_params,
        data_paths=data_paths,
        device=device,
        world_size=world_size,
        rank=rank,
    )
    trainer.fit()
    if dist.is_initialized():
        distributed_barrier()
        dist.destroy_process_group()


def _build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "llama":
        cls = LlamaForCausalLMTpSp if args.use_sp else LlamaForCausalLMTp if args.tp_size > 1 else LlamaForCausalLM
        return cls(
            LlamaConfig(
                vocab_size=args.vocab_size,
                hidden_size=args.dim,
                intermediate_size=args.hidden_size,
                num_hidden_layers=args.n_layers,
                num_attention_heads=args.n_heads,
                num_key_value_heads=args.n_kv_heads or args.n_heads,
                max_position_embeddings=args.seq_len,
                rms_norm_eps=args.eps,
            )
        )
    cls = TinyTransformerTpSp if args.use_sp else TinyTransformerTp if args.tp_size > 1 else TinyTransformer
    return cls(
        dim=args.dim,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads or args.n_heads,
        hidden_size=args.hidden_size,
        eps=args.eps,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
    )


def _build_runtime(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> RuntimeCore:
    plugins = []
    optimizer_factory = None
    if args.tp_size > 1:
        plugins.append(TensorParallelPlugin())
    if args.use_sp:
        plugins.append(SequenceParallelPlugin())
    if args.zero_stage == 0:
        if args.dp_size > 1:
            plugins.append(_build_ddp(args.ddp_mode or "sync"))
        optimizer_factory = lambda module: torch.optim.AdamW(module.parameters(), lr=args.lr)
    elif args.zero_stage == 1:
        plugins.append(Zero1Plugin(optimizer_cls=torch.optim.AdamW, lr=args.lr))
    elif args.zero_stage == 2:
        plugins.append(Zero2Plugin(optimizer_cls=torch.optim.AdamW, lr=args.lr))
    else:
        plugins.append(
            Zero3Plugin(
                wrap_cls={torch.nn.Linear, ColumnParallelLinear, RowParallelLinear},
                optimizer_cls=torch.optim.AdamW,
                lr=args.lr,
            )
        )
    compute_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    if compute_dtype is not None:
        plugins.append(PrecisionPlugin(compute_dtype=compute_dtype))
    if args.grad_clip is not None:
        plugins.append(GradClipPlugin(max_norm=args.grad_clip))
    if not args.disable_perf_metrics:
        plugins.append(PerfMetricsPlugin())

    return RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=1, cp=1, ep=1),
        plan=ParallelPlan(zero_stage=args.zero_stage),
        device=device,
        model=model,
        optimizer_factory=optimizer_factory,
        plugins=plugins,
        grad_accum_steps=args.grad_accum_steps,
    )


def _build_ddp(mode: str) -> DataParallelPlugin | BucketDataParallelPlugin:
    if mode == "bucket":
        return BucketDataParallelPlugin()
    return DataParallelPlugin(async_op=(mode == "async"))


def _build_logging(args: argparse.Namespace, rank: int) -> tuple[list[MetricLogger] | None, WandbCheckpointUploader | None]:
    if rank != 0:
        return None, None
    loggers: list[MetricLogger] = [ConsoleMetricLogger()]
    wandb_logger: WandbMetricLogger | None = None
    if args.metrics_jsonl is not None:
        loggers.append(JsonlMetricLogger(args.metrics_jsonl))
    if args.wandb_project is not None and args.wandb_mode != "disabled":
        wandb_logger = WandbMetricLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            run_id=args.wandb_run_id,
            mode=args.wandb_mode,
            tags=_parse_tags(args.wandb_tags),
            config=vars(args),
        )
        loggers.append(wandb_logger)
    checkpoint_uploader = None
    if args.wandb_checkpoint_every is not None:
        if wandb_logger is None:
            raise ValueError("--wandb-checkpoint-every requires enabled W&B logging")
        checkpoint_uploader = WandbCheckpointUploader(
            wandb_logger,
            every_steps=args.wandb_checkpoint_every,
            artifact_prefix=args.wandb_run_name,
        )
    return loggers, checkpoint_uploader


def _preparse_config_path() -> str | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()
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


def _config_key_to_arg_dest(section: str, key: str) -> str:
    aliases = {
        ("data", "paths"): "data",
        ("data", "seq_len"): "seq_len",
        ("data", "micro_batch_size"): "micro_batch_size",
        ("model", "type"): "model",
        ("model", "hidden_size"): "dim",
        ("model", "dim"): "dim",
        ("model", "intermediate_size"): "hidden_size",
        ("model", "num_layers"): "n_layers",
        ("model", "n_layers"): "n_layers",
        ("model", "num_heads"): "n_heads",
        ("model", "n_heads"): "n_heads",
        ("model", "num_kv_heads"): "n_kv_heads",
        ("model", "n_kv_heads"): "n_kv_heads",
        ("model", "vocab_size"): "vocab_size",
        ("model", "rms_norm_eps"): "eps",
        ("model", "eps"): "eps",
        ("parallel", "dp_size"): "dp_size",
        ("parallel", "tp_size"): "tp_size",
        ("parallel", "use_sp"): "use_sp",
        ("parallel", "zero_stage"): "zero_stage",
        ("parallel", "ddp_mode"): "ddp_mode",
        ("training", "max_steps"): "max_steps",
        ("training", "grad_accum_steps"): "grad_accum_steps",
        ("training", "lr"): "lr",
        ("training", "precision"): "precision",
        ("training", "grad_clip"): "grad_clip",
        ("training", "disable_perf_metrics"): "disable_perf_metrics",
        ("logging", "log_every"): "log_every",
        ("logging", "metrics_jsonl"): "metrics_jsonl",
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
    }
    return aliases.get((section, key), key)


def _parse_tags(tags: str | list[str] | None) -> list[str] | None:
    if tags is None:
        return None
    if isinstance(tags, list):
        parsed = [str(tag).strip() for tag in tags if str(tag).strip()]
        return parsed or None
    parsed = [tag.strip() for tag in tags.split(",") if tag.strip()]
    return parsed or None


def _print_run_summary(
    *,
    args: argparse.Namespace,
    runtime: RuntimeCore,
    initial_trainable_params: int,
    data_paths: list[Path],
    device: torch.device,
    world_size: int,
    rank: int,
) -> None:
    if rank != 0:
        return
    local_trainable_params = _count_trainable_params(runtime.model)
    global_batch_tokens = args.micro_batch_size * args.seq_len * args.dp_size * args.grad_accum_steps
    total_train_tokens = global_batch_tokens * args.max_steps
    plugin_names = [plugin.name for plugin in runtime.plugins]
    flops_per_token = runtime.state.static_metrics.get("perf/flops_per_token")
    print("=== pretrain run ===")
    print(f"config={args.config}")
    print(f"model={args.model} initial_trainable_params={initial_trainable_params:,}")
    print(f"runtime_local_trainable_params={local_trainable_params:,}")
    print(
        "mesh="
        f"dp={args.dp_size} tp={args.tp_size} pp=1 cp=1 "
        f"world_size={world_size} device={device}"
    )
    print(f"plugins={plugin_names}")
    print(
        "training="
        f"precision={args.precision} lr={args.lr} grad_accum_steps={args.grad_accum_steps} "
        f"micro_batch_size={args.micro_batch_size} seq_len={args.seq_len}"
    )
    print(f"tokens_per_step={global_batch_tokens:,} target_tokens={total_train_tokens:,}")
    print(
        "performance="
        f"flops_per_token={_format_optional_number(flops_per_token)}"
    )
    print(f"data_shards={len(data_paths)} first_data={data_paths[0]}")
    print(
        "logging="
        f"log_every={args.log_every} jsonl={args.metrics_jsonl} "
        f"wandb_project={args.wandb_project} wandb_mode={args.wandb_mode} "
        f"wandb_run_id={args.wandb_run_id} "
        f"wandb_checkpoint_every={args.wandb_checkpoint_every}"
    )
    print(
        "checkpoint="
        f"dir={args.checkpoint_dir} every={args.checkpoint_every} "
        f"keep_last={args.checkpoint_keep_last} keep_every_n_steps={args.checkpoint_keep_every_n_steps} "
        f"min_free_gb={args.checkpoint_min_free_gb} resume_from={args.resume_from}"
    )


def _count_trainable_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _format_optional_number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    return "None"


def _expand_data_paths(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.bin")))
        else:
            paths.append(path)
    if not paths:
        raise ValueError("no token shard paths found")
    return paths


def _maybe_init_distributed(args: argparse.Namespace) -> None:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world_size <= 1 or dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", args.master_port)
    backend = args.backend or ("nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend=backend)


def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


if __name__ == "__main__":
    main()
