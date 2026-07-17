from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data import PackedSFTDataset, PretrainingDataLoader, SFTDataLoader, TokenShardDataset
from models import (
    ActivationCheckpointConfig,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaForCausalLMTp,
    LlamaForCausalLMTpSp,
    OlmoConfig,
    OlmoForCausalLM,
    OlmoForCausalLMTp,
    OlmoForCausalLMTpSp,
    OlmoRMSNorm,
    TinyMoETransformer,
    TinyMoETransformerTp,
    TinyMoETransformerTpSp,
    TinyTransformer,
    TinyTransformerTp,
    TinyTransformerTpSp,
)
from models.llama import LlamaRMSNorm
from models.tiny_transformer import RmsNorm
from parallel import ParallelPlan
from parallel.context_interfaces import ContextParallelAttentionCoreType
from parallel.plan import PipelineScheduleConfig
from runtime import MeshConfig, RuntimeCore
from runtime.layers.distributed_rmsnorm import DistributedRMSNorm
from runtime.plugins.ddp import BucketDataParallelPlugin, DataParallelPlugin
from runtime.plugins.cp import ContextParallelPlugin
from runtime.plugins.ep import ExpertParallelPlugin
from runtime.plugins.grad_clip import GradClipPlugin
from runtime.plugins.metrics import MetricPlugin
from runtime.plugins.pp import PipelineParallelPlugin
from runtime.plugins.fp16 import Fp16Plugin
from runtime.plugins.sp import SequenceParallelPlugin
from runtime.plugins.torch_profiler import TorchProfilerPlugin
from runtime.plugins.tp import TensorParallelPlugin
from runtime.plugins.zero1 import Zero1Plugin
from runtime.plugins.zero2 import Zero2Plugin
from runtime.plugins.zero3 import Zero3Plugin
from train import Trainer, TrainerConfig
from utils.attention_backend import ATTENTION_BACKEND_CHOICES, AttentionBackend
from utils.metrics import (
    ConsoleMetricLogger,
    JsonlMetricLogger,
    MetricLogger,
    WandbCheckpointUploader,
    WandbMetricLogger,
)
from utils.distributed import distributed_barrier


_ZERO3_WRAP_CLS = {
    torch.nn.Linear,
    torch.nn.Embedding,
    torch.nn.LayerNorm,
    RmsNorm,
    LlamaRMSNorm,
    OlmoRMSNorm,
    DistributedRMSNorm,
}

def parse_args() -> argparse.Namespace:
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
    expected_world_size = args.dp_size * args.pp_size * args.cp_size * args.tp_size
    if world_size != expected_world_size:
        raise ValueError(
            f"world_size={world_size} must equal dp_size * pp_size * cp_size * tp_size={expected_world_size}"
        )
    if args.use_sp and args.tp_size <= 1:
        raise ValueError("--use-sp requires --tp-size > 1")
    if args.cp_size > 1 and args.seq_len % args.cp_size != 0:
        raise ValueError("--cp-size requires --seq-len divisible by cp_size for CP v0")
    if args.ep_size > 1 and args.dp_size % args.ep_size != 0:
        raise ValueError("--ep-size must divide --dp-size")
    if args.ep_size > 1 and args.model != "tiny_moe":
        raise ValueError("--ep-size > 1 requires --model tiny_moe")
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
    dp_rank = rank // (args.pp_size * args.cp_size * args.tp_size)
    data_paths, loader, data_format = _build_dataloader(args, dp_rank=dp_rank)
    model = _build_model(args)
    initial_trainable_params = _count_trainable_params(model)
    runtime = _build_runtime(args, model, device)
    logger, checkpoint_uploader = (None, None) if args.dry_run else _build_logging(args, rank)
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
        data_format=data_format,
        device=device,
        world_size=world_size,
        rank=rank,
    )
    if args.run_manifest is not None:
        _write_run_manifest(
            args=args,
            runtime=runtime,
            initial_trainable_params=initial_trainable_params,
            data_paths=data_paths,
            data_format=data_format,
            device=device,
            world_size=world_size,
            rank=rank,
        )
    if args.dry_run:
        if rank == 0:
            print("dry_run=true")
        runtime.close()
        if dist.is_initialized():
            distributed_barrier()
            dist.destroy_process_group()
        return
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
                attention_backend=args.attention_backend,
                activation_checkpointing=ActivationCheckpointConfig(
                    enabled=args.activation_checkpointing,
                    every_n_layers=args.activation_checkpoint_every_n_layers,
                ),
            )
        )
    if args.model in {"olmo", "olmo2"}:
        cls = OlmoForCausalLMTpSp if args.use_sp else OlmoForCausalLMTp if args.tp_size > 1 else OlmoForCausalLM
        return cls(
            OlmoConfig(
                vocab_size=args.vocab_size,
                hidden_size=args.dim,
                intermediate_size=args.hidden_size,
                num_hidden_layers=args.n_layers,
                num_attention_heads=args.n_heads,
                num_key_value_heads=args.n_kv_heads or args.n_heads,
                max_position_embeddings=args.seq_len,
                rms_norm_eps=args.eps,
                attention_backend=args.attention_backend,
                activation_checkpointing=ActivationCheckpointConfig(
                    enabled=args.activation_checkpointing,
                    every_n_layers=args.activation_checkpoint_every_n_layers,
                ),
            )
        )
    if args.model == "tiny_moe":
        cls = TinyMoETransformerTpSp if args.use_sp else TinyMoETransformerTp if args.tp_size > 1 else TinyMoETransformer
        return cls(
            dim=args.dim,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads or args.n_heads,
            hidden_size=args.hidden_size,
            eps=args.eps,
            n_layers=args.n_layers,
            vocab_size=args.vocab_size,
            max_seq_len=args.seq_len,
            num_experts=args.num_experts,
            attention_backend=args.attention_backend,
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
        attention_backend=args.attention_backend,
    )


def _build_runtime(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> RuntimeCore:
    plugins = []
    grad_clip_max_norm = None
    optimizer_factory = _build_optimizer_factory(args)
    scheduler_factory = _build_scheduler_factory(args)
    if args.tp_size > 1:
        plugins.append(TensorParallelPlugin())
    if args.use_sp:
        plugins.append(SequenceParallelPlugin())
    if args.cp_size > 1:
        plugins.append(ContextParallelPlugin())
    if args.ep_size > 1:
        plugins.append(ExpertParallelPlugin())
    if args.pp_size > 1:
        plugins.append(PipelineParallelPlugin())
    if args.zero_stage == 0:
        if args.dp_size > 1:
            plugins.append(_build_ddp(args.ddp_mode or "sync"))
    elif args.zero_stage == 1:
        plugins.append(Zero1Plugin())
    elif args.zero_stage == 2:
        plugins.append(Zero2Plugin())
    else:
        plugins.append(
            Zero3Plugin(
                wrap_cls=_ZERO3_WRAP_CLS,
            )
        )
    runtime_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    if runtime_dtype == torch.float16:
        plugins.append(Fp16Plugin())
    if args.grad_clip is not None:
        if args.zero_stage == 0:
            plugins.append(GradClipPlugin(max_norm=args.grad_clip))
        else:
            grad_clip_max_norm = args.grad_clip
    if not args.disable_metrics:
        plugins.append(MetricPlugin())
    if args.torch_profiler:
        plugins.append(
            TorchProfilerPlugin(
                trace_dir=args.torch_profiler_dir,
                wait=args.torch_profiler_wait,
                warmup=args.torch_profiler_warmup,
                active=args.torch_profiler_active,
                repeat=args.torch_profiler_repeat,
                record_shapes=args.torch_profiler_record_shapes,
                profile_memory=args.torch_profiler_profile_memory,
                with_stack=args.torch_profiler_with_stack,
                with_flops=args.torch_profiler_with_flops,
                rank0_only=args.torch_profiler_rank0_only,
            )
        )

    return RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=args.pp_size, cp=args.cp_size, ep=args.ep_size),
        plan=ParallelPlan(
            cp_attn_core=ContextParallelAttentionCoreType(args.cp_attn_core),
            pp_schedule=PipelineScheduleConfig(microbatches=args.pp_microbatches),
        ),
        device=device,
        dtype=runtime_dtype,
        model=model,
        grad_clip_max_norm=grad_clip_max_norm,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        plugins=plugins,
    )


def _build_ddp(mode: str) -> DataParallelPlugin | BucketDataParallelPlugin:
    if mode == "bucket":
        return BucketDataParallelPlugin()
    return DataParallelPlugin(async_op=(mode == "async"))


def _build_optimizer_factory(args: argparse.Namespace):
    if args.lr < 0:
        raise ValueError("--lr must be >= 0")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0")
    if not 0.0 <= args.adam_beta1 < 1.0:
        raise ValueError("--adam-beta1 must be in [0, 1)")
    if not 0.0 <= args.adam_beta2 < 1.0:
        raise ValueError("--adam-beta2 must be in [0, 1)")
    if args.adam_eps <= 0:
        raise ValueError("--adam-eps must be > 0")

    def build_optimizer(params) -> torch.optim.Optimizer:
        kwargs = {
            "lr": args.lr,
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_eps,
            "weight_decay": args.weight_decay,
        }
        if args.fused_adamw:
            kwargs["fused"] = True
        return torch.optim.AdamW(params, **kwargs)

    return build_optimizer


def _build_scheduler_factory(args: argparse.Namespace):
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.min_lr < 0:
        raise ValueError("--min-lr must be >= 0")
    if args.min_lr > args.lr:
        raise ValueError("--min-lr must be <= --lr")
    if args.lr_schedule == "constant" and args.warmup_steps == 0:
        return None

    min_lr_ratio = args.min_lr / args.lr if args.lr > 0 else 0.0

    def lr_multiplier(step: int) -> float:
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return max(0.0, step / args.warmup_steps)
        if args.lr_schedule == "constant":
            return 1.0
        decay_steps = max(1, args.max_steps - args.warmup_steps)
        progress = min(1.0, max(0.0, (step - args.warmup_steps) / decay_steps))
        if args.lr_schedule == "linear":
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
        if args.lr_schedule == "cosine":
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(progress * math.pi))
        raise ValueError(f"unknown lr_schedule={args.lr_schedule}")

    return lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


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
    data_format: str,
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
    print("=== train run ===")
    print(f"config={args.config}")
    print(f"model={args.model} initial_trainable_params={initial_trainable_params:,}")
    print(f"runtime_local_trainable_params={local_trainable_params:,}")
    print(
        "mesh="
        f"dp={args.dp_size} tp={args.tp_size} pp={args.pp_size} "
        f"cp={args.cp_size} ep={args.ep_size} "
        f"world_size={world_size} device={device}"
    )
    print(f"plugins={plugin_names}")
    print(
        "training="
        f"dry_run={args.dry_run} "
        f"precision={args.precision} lr={args.lr} weight_decay={args.weight_decay} "
        f"adam_betas=({args.adam_beta1}, {args.adam_beta2}) adam_eps={args.adam_eps} "
        f"fused_adamw={args.fused_adamw} "
        f"lr_schedule={args.lr_schedule} "
        f"warmup_steps={args.warmup_steps} min_lr={args.min_lr} grad_accum_steps={args.grad_accum_steps} "
        f"micro_batch_size={args.micro_batch_size} seq_len={args.seq_len} "
        f"pp_microbatches={args.pp_microbatches}"
    )
    print(
        "model_features="
        f"attention_backend={args.attention_backend} "
        f"activation_checkpointing={args.activation_checkpointing} "
        f"activation_checkpoint_every_n_layers={args.activation_checkpoint_every_n_layers}"
    )
    print(f"tokens_per_step={global_batch_tokens:,} target_tokens={total_train_tokens:,}")
    print(
        "performance="
        f"flops_per_token={_format_optional_number(flops_per_token)}"
    )
    print(f"data_format={data_format} data_shards={len(data_paths)} first_data={data_paths[0]}")
    print(
        "logging="
        f"log_every={args.log_every} jsonl={args.metrics_jsonl} "
        f"run_manifest={args.run_manifest} "
        f"wandb_project={args.wandb_project} wandb_mode={args.wandb_mode} "
        f"wandb_run_id={args.wandb_run_id} "
        f"wandb_checkpoint_every={args.wandb_checkpoint_every}"
    )
    print(
        "profiling="
        f"torch_profiler={args.torch_profiler} dir={args.torch_profiler_dir} "
        f"schedule=(wait={args.torch_profiler_wait}, warmup={args.torch_profiler_warmup}, "
        f"active={args.torch_profiler_active}, repeat={args.torch_profiler_repeat}) "
        f"rank0_only={args.torch_profiler_rank0_only}"
    )
    print(
        "checkpoint="
        f"dir={args.checkpoint_dir} every={args.checkpoint_every} "
        f"keep_last={args.checkpoint_keep_last} keep_every_n_steps={args.checkpoint_keep_every_n_steps} "
        f"min_free_gb={args.checkpoint_min_free_gb} resume_from={args.resume_from}"
    )


def _write_run_manifest(
    *,
    args: argparse.Namespace,
    runtime: RuntimeCore,
    initial_trainable_params: int,
    data_paths: list[Path],
    data_format: str,
    device: torch.device,
    world_size: int,
    rank: int,
) -> None:
    if rank != 0:
        return
    path = Path(args.run_manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_run_manifest(
        args=args,
        runtime=runtime,
        initial_trainable_params=initial_trainable_params,
        data_paths=data_paths,
        data_format=data_format,
        device=device,
        world_size=world_size,
    )
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def _build_run_manifest(
    *,
    args: argparse.Namespace,
    runtime: RuntimeCore,
    initial_trainable_params: int,
    data_paths: list[Path],
    data_format: str,
    device: torch.device,
    world_size: int,
) -> dict[str, Any]:
    local_trainable_params = _count_trainable_params(runtime.model)
    global_batch_tokens = args.micro_batch_size * args.seq_len * args.dp_size * args.grad_accum_steps
    total_train_tokens = global_batch_tokens * args.max_steps
    optimizer, scheduler = runtime.get_optimizer_and_scheduler()
    return {
        "version": 1,
        "git": {
            "commit": _git_output(["rev-parse", "HEAD"]),
            "branch": _git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": _git_is_dirty(),
        },
        "args": vars(args),
        "resolved": {
            "device": str(device),
            "world_size": world_size,
            "mesh": {
                "dp": args.dp_size,
                "tp": args.tp_size,
                "pp": args.pp_size,
                "cp": args.cp_size,
                "ep": args.ep_size,
            },
            "plugins": [
                {
                    "id": plugin.id.value,
                    "name": plugin.name,
                    "owns_optimizer": plugin.owns_optimizer,
                    "owns_step_runner": plugin.owns_step_runner,
                }
                for plugin in runtime.plugins
            ],
            "model": {
                "type": args.model,
                "initial_trainable_params": initial_trainable_params,
                "runtime_local_trainable_params": local_trainable_params,
                "attention_backend": args.attention_backend,
                "activation_checkpointing": args.activation_checkpointing,
                "activation_checkpoint_every_n_layers": args.activation_checkpoint_every_n_layers,
            },
            "optimizer": {
                "type": type(optimizer).__name__ if optimizer is not None else None,
                "scheduler": type(scheduler).__name__ if scheduler is not None else None,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "adam_beta1": args.adam_beta1,
                "adam_beta2": args.adam_beta2,
                "adam_eps": args.adam_eps,
                "fused_adamw": args.fused_adamw,
                "fused_adamw_applied": optimizer.defaults.get("fused") if optimizer is not None else None,
                "lr_schedule": args.lr_schedule,
                "warmup_steps": args.warmup_steps,
                "min_lr": args.min_lr,
            },
            "training": {
                "dry_run": args.dry_run,
                "precision": args.precision,
                "grad_accum_steps": args.grad_accum_steps,
                "micro_batch_size": args.micro_batch_size,
                "seq_len": args.seq_len,
                "tokens_per_step": global_batch_tokens,
                "target_tokens": total_train_tokens,
                "max_steps": args.max_steps,
                "grad_clip": args.grad_clip,
            },
            "data": {
                "format": data_format,
                "num_shards": len(data_paths),
                "paths": [str(path) for path in data_paths],
            },
            "logging": {
                "log_every": args.log_every,
                "metrics_jsonl": args.metrics_jsonl,
                "wandb_project": args.wandb_project,
                "wandb_run_name": args.wandb_run_name,
                "wandb_run_id": args.wandb_run_id,
                "wandb_mode": args.wandb_mode,
                "wandb_checkpoint_every": args.wandb_checkpoint_every,
            },
            "profiling": {
                "torch_profiler": args.torch_profiler,
                "torch_profiler_dir": args.torch_profiler_dir,
                "torch_profiler_wait": args.torch_profiler_wait,
                "torch_profiler_warmup": args.torch_profiler_warmup,
                "torch_profiler_active": args.torch_profiler_active,
                "torch_profiler_repeat": args.torch_profiler_repeat,
                "torch_profiler_record_shapes": args.torch_profiler_record_shapes,
                "torch_profiler_profile_memory": args.torch_profiler_profile_memory,
                "torch_profiler_with_stack": args.torch_profiler_with_stack,
                "torch_profiler_with_flops": args.torch_profiler_with_flops,
                "torch_profiler_rank0_only": args.torch_profiler_rank0_only,
            },
            "checkpoint": {
                "dir": args.checkpoint_dir,
                "every": args.checkpoint_every,
                "keep_last": args.checkpoint_keep_last,
                "keep_every_n_steps": args.checkpoint_keep_every_n_steps,
                "min_free_gb": args.checkpoint_min_free_gb,
                "resume_from": args.resume_from,
            },
        },
    }


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_is_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


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


def _build_dataloader(args: argparse.Namespace, *, dp_rank: int):
    data_format = _infer_data_format(args.data, explicit=args.data_format)
    if data_format == "pretrain":
        data_paths = _expand_data_paths(args.data)
        loader = PretrainingDataLoader(
            TokenShardDataset(data_paths, dtype=np.dtype(args.token_dtype)),
            seq_len=args.seq_len,
            micro_batch_size=args.micro_batch_size,
            dp_rank=dp_rank,
            dp_world_size=args.dp_size,
            seed=args.seed,
        )
        return data_paths, loader, data_format

    data_source = _resolve_sft_data_source(args.data)
    dataset = PackedSFTDataset(data_source)
    if args.seq_len != dataset.seq_len:
        raise ValueError(
            f"SFT dataset seq_len={dataset.seq_len} must match --seq-len={args.seq_len}"
        )
    loader = SFTDataLoader(
        dataset,
        micro_batch_size=args.micro_batch_size,
        dp_rank=dp_rank,
        dp_world_size=args.dp_size,
        seed=args.seed,
    )
    return list(dataset.shard_paths), loader, data_format


def _infer_data_format(items: list[str], *, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    if len(items) != 1:
        return "pretrain"
    candidate = Path(items[0])
    meta_path = candidate / "meta.json" if candidate.is_dir() else candidate
    if meta_path.is_file() and meta_path.name == "meta.json":
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "pretrain"
        if payload.get("format") == "maltos_sft_packed":
            return "sft"
    return "pretrain"


def _resolve_sft_data_source(items: list[str]) -> Path:
    if len(items) != 1:
        raise ValueError("--data-format sft expects a single SFT dataset directory or meta.json path")
    return Path(items[0])


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
