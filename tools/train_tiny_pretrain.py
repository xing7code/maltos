from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from data import PretrainingDataLoader, TokenShardDataset
from models import TinyTransformer, TinyTransformerTp, TinyTransformerTpSp
from parallel import ParallelPlan
from runtime import MeshConfig, RuntimeCore
from runtime.layers.tp import ColumnParallelLinear, RowParallelLinear
from runtime.plugins.ddp import BucketDataParallelPlugin, DataParallelPlugin
from runtime.plugins.grad_clip import GradClipPlugin
from runtime.plugins.precision import PrecisionPlugin
from runtime.plugins.sp import SequenceParallelPlugin
from runtime.plugins.tp import TensorParallelPlugin
from runtime.plugins.zero1 import Zero1Plugin
from runtime.plugins.zero2 import Zero2Plugin
from runtime.plugins.zero3 import Zero3Plugin
from train import Trainer, TrainerConfig
from utils.metrics import ConsoleMetricLogger, JsonlMetricLogger, MetricLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny Transformer pretraining recipe.")
    parser.add_argument("--data", type=str, nargs="+", required=True, help="Token shard .bin files or directories")
    parser.add_argument("--token-dtype", type=str, default="uint32", choices=("uint16", "uint32", "int64"))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

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
    parser.add_argument("--use-sp", action="store_true")
    parser.add_argument("--ddp-mode", type=str, default="sync", choices=("sync", "async", "bucket"))
    parser.add_argument("--precision", type=str, default="fp32", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--grad-clip", type=float, default=None)

    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--metrics-jsonl", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--resume-from", type=str, default=None)

    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="29550")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _maybe_init_distributed(args)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size != args.dp_size * args.tp_size:
        raise ValueError(f"world_size={world_size} must equal dp_size * tp_size={args.dp_size * args.tp_size}")
    if args.use_sp and args.tp_size <= 1:
        raise ValueError("--use-sp requires --tp-size > 1")
    if args.zero_stage > 0 and args.dp_size <= 1:
        raise ValueError("--zero-stage > 0 requires --dp-size > 1")

    torch.manual_seed(args.seed)
    dp_rank = rank // args.tp_size
    model = _build_model(args)
    loader = PretrainingDataLoader(
        TokenShardDataset(_expand_data_paths(args.data), dtype=np.dtype(args.token_dtype)),
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        dp_rank=dp_rank,
        dp_world_size=args.dp_size,
        seed=args.seed,
    )
    runtime = _build_runtime(args, model)
    logger = _build_logger(args, rank)
    trainer = Trainer(
        runtime=runtime,
        dataloader=loader,
        config=TrainerConfig(
            max_steps=args.max_steps,
            log_every=args.log_every,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume_from,
        ),
        logger=logger,
    )
    trainer.setup()
    trainer.fit()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _build_model(args: argparse.Namespace) -> TinyTransformer:
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


def _build_runtime(args: argparse.Namespace, model: TinyTransformer) -> RuntimeCore:
    plugins = []
    optimizer = None
    if args.tp_size > 1:
        plugins.append(TensorParallelPlugin())
    if args.use_sp:
        plugins.append(SequenceParallelPlugin())
    if args.zero_stage == 0:
        if args.dp_size > 1:
            plugins.append(_build_ddp(args.ddp_mode))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
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

    return RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=1, cp=1, ep=1),
        plan=ParallelPlan(zero_stage=args.zero_stage),
        model=model,
        optimizer=optimizer,
        plugins=plugins,
        grad_accum_steps=args.grad_accum_steps,
    )


def _build_ddp(mode: str) -> DataParallelPlugin | BucketDataParallelPlugin:
    if mode == "bucket":
        return BucketDataParallelPlugin()
    return DataParallelPlugin(async_op=(mode == "async"))


def _build_logger(args: argparse.Namespace, rank: int) -> MetricLogger | None:
    if rank != 0:
        return None
    if args.metrics_jsonl is not None:
        return JsonlMetricLogger(args.metrics_jsonl)
    return ConsoleMetricLogger()


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


if __name__ == "__main__":
    main()
