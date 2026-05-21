"""Smoke test for TinyTransformer with the Trainer loop via mp.spawn.

Single process:
  PYTHONPATH=. .venv/bin/python train_system/tests/smoke_tiny_transformer.py --world-size 1

Multi-process (CPU/gloo):

TP:
  PYTHONPATH=. .venv/bin/python train_system/tests/smoke_tiny_transformer.py --world-size 2 --tp-size 2 --model-class tiny_tp --v 3

TP+SP:
  PYTHONPATH=. .venv/bin/python train_system/tests/smoke_tiny_transformer.py --world-size 2 --tp-size 2 --model-class tiny_tp_sp --use-sp --v 3

DDP:
  PYTHONPATH=. .venv/bin/python train_system/tests/smoke_tiny_transformer.py --world-size 2 --ddp-size 2 --ddp-type naive --model-class tiny --v 3

DDP + TP + SP:
  PYTHONPATH=. .venv/bin/python train_system/tests/smoke_tiny_transformer.py --world-size 4 --ddp-size 2 --ddp-type naive --tp-size 2 --use-sp --model-class tiny_tp_sp --v 3
"""

from __future__ import annotations

import argparse
import os
from absl import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train_system.engine import Trainer
from train_system.examples import TinyTransformer, TinyTransformerTp, TinyTransformerTpSp, CausalSelfAttention, MLP
from train_system.parallel import ParallelPlan
from train_system.runtime import MeshAxis, MeshConfig, ProcessGroupManager, RuntimeContext
from train_system.runtime.plugins.tp import TpPlugin
from train_system.runtime.plugins.sp import SpPlugin
from train_system.runtime.plugins.ddp import DdpWithBucketPlugin, NaiveDdpPlugin, NaiveAsyncDdpPlugin
from train_system.runtime.plugins.zero1 import Zero1Plugin
from train_system.runtime.plugins.zero2 import Zero2Plugin
from train_system.runtime.plugins.zero3 import Zero3Plugin


_REGISTRY_MODLE_CLASS = {
    "tiny": TinyTransformer,
    "tiny_tp": TinyTransformerTp,
    "tiny_tp_sp": TinyTransformerTpSp,
}

_REGISTRY_DDP_PLUGIN = {
    "naive": NaiveDdpPlugin,
    "naive_async": NaiveAsyncDdpPlugin,
    "bucket_async": DdpWithBucketPlugin,
    "zero1": Zero1Plugin,
    "zero2": Zero2Plugin,
    "zero3": Zero3Plugin,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyTransformer smoke test with mp.spawn")
    parser.add_argument("--world-size", type=int, default=2, help="Number of local processes")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1", help="Master address for c10d init")
    parser.add_argument("--master-port", type=int, default=29501, help="Master port for c10d init")
    parser.add_argument("--backend", type=str, default="auto", help="Process group backend")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor Parallel size to test.")
    parser.add_argument("--use-sp", action="store_true", default=False, help="Whether to use sp along with tp.")
    parser.add_argument("--ddp-size", type=int, default=1, help="Distributed Data Parallel size to test.")
    parser.add_argument("--ddp-type", type=str, default="naive", help="The type of DDP plugin.")
    parser.add_argument("--ddp-bucket-mb-size", type=int, default=25, help="The bucket size in MB for DDP plugin.")
    parser.add_argument("--model-class", type=str, default="tiny", help="The model class to test on.")
    parser.add_argument("--steps", type=int, default=3, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size")
    parser.add_argument("-v", "--v", type=int, default=0, help="Verbosity level for VLOG")
    parser.add_argument("--profile", action="store_true", default=False, help="whether to do profile.")
    parser.add_argument("--profile-dir", type=str, default="./profile_logs", help="where to save the profile logs.")
    return parser.parse_args()


def _get_device(rank: int) -> torch.device:
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        assert rank < n_gpus, f"rank={rank} >= n_gpus={n_gpus}"
        torch.cuda.set_device(rank)
        return torch.device("cuda", rank)
    return torch.device("cpu")


def _data_iter(vocab_size: int, batch_size: int, seq_len: int, device: torch.device):
    while True:
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device)
        yield (tokens, tokens.clone())


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    logging.set_verbosity(args.v)
    logging.use_absl_handler()
    backend = args.backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if args.world_size > 1:
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{args.master_addr}:{args.master_port}",
            rank=rank,
            world_size=args.world_size,
        )
    device = _get_device(rank)
    mesh=MeshConfig(dp=args.ddp_size, tp=args.tp_size, pp=1, cp=1, ep=1)
    plan = ParallelPlan()
    group_manager = ProcessGroupManager.from_mesh(mesh)
    plugins = []
    if args.tp_size > 1:
        tp_group = group_manager.get_group(MeshAxis.TP)
        plugins += [TpPlugin(tp_group)]
        if args.use_sp:
            plugins += [SpPlugin(tp_group)]
    if args.ddp_size > 1:
        dp_group = group_manager.get_group(MeshAxis.DP)
        if "bucket" in args.ddp_type:
            plugins += [_REGISTRY_DDP_PLUGIN[args.ddp_type](dp_group, args.ddp_bucket_mb_size)]
        elif args.ddp_type in ("zero1", "zero2"):
            plugins += [_REGISTRY_DDP_PLUGIN[args.ddp_type](dp_group, args.ddp_bucket_mb_size, torch.optim.AdamW, lr=1e-3)]
        elif args.ddp_type == "zero3":
            plugins += [_REGISTRY_DDP_PLUGIN[args.ddp_type](dp_group, {CausalSelfAttention, MLP}, torch.optim.AdamW, lr=1e-3)]
        else:
            plugins += [_REGISTRY_DDP_PLUGIN[args.ddp_type](dp_group)]

    ctx = RuntimeContext(plan=plan, plugins=plugins)

    model = _REGISTRY_MODLE_CLASS[args.model_class](
        dim=64,
        n_heads=4,
        n_kv_heads=4,
        hidden_size=128,
        eps=1e-5,
        n_layers=2,
        vocab_size=256,
        max_seq_len=64,
    ).to(device)
    trainer = Trainer(context=ctx, model=model, optimizer=None)
    trainer.setup()
    plugin_owns_optimizer = any(getattr(p, "owns_optimizer", False) for p in ctx.plugins)
    if not plugin_owns_optimizer:
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)

    if args.ddp_size > 1:
        assert args.batch_size % args.ddp_size==0, f"batch size {args.batch_size} must be divisible by ddp_size {args.ddp_size}!"
        local_batch_size = args.batch_size // args.ddp_size
    else:
        local_batch_size = args.batch_size

    if args.profile and rank==0:
        from torch.profiler import profile, ProfilerActivity, schedule
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        prof = profile(
            activities=activities,
            schedule=schedule(wait=0, warmup=0, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile_dir),
            record_shapes=True,
            with_flops=True,
        )
        prof.start()
    else:
        prof = None
    trainer.train_steps(
        _data_iter(vocab_size=args.vocab_size, batch_size=local_batch_size, seq_len=args.seq_len, device=device),
        steps=args.steps,
        profiler=prof,
    )
    if prof is not None:
        prof.stop()
        print(f"[rank={rank}] profiler stopped, check {args.profile_dir}", flush=True)
        import os
        print(f"[rank={rank}] profile_logs contents: {os.listdir(args.profile_dir)}", flush=True)
    logging.info(f"[rank={rank}] tiny transformer smoke ok")
    if args.world_size > 1:
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    assert args.world_size >= 1 and args.tp_size >= 1 and args.ddp_size >= 1 and args.tp_size * args.ddp_size == args.world_size, f"Invalid args config on tp_size, ddp_size, world_size. {args}"
    assert args.model_class in _REGISTRY_MODLE_CLASS, f"invalid model_class {args.model_class}"

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
