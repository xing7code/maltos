"""Equivalence test: TinyTransformer (baseline) vs TinyTransformerTpSp (TP+SP).

Both should produce the same loss given the same weights and data.

Usage:

baseline vs TP only:
  PYTHONPATH=. .venv/bin/python train_system/tests/tiny_transformer_tp_equivalence.py --world-size 2 --tp-size 2

baseline vs TP+SP:
  PYTHONPATH=. .venv/bin/python train_system/tests/tiny_transformer_tp_equivalence.py --world-size 2 --tp-size 2 --use-sp true
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train_system.engine import Trainer
from train_system.examples import TinyTransformer, TinyTransformerTp, TinyTransformerTpSp
from train_system.parallel import ParallelPlan, ProcessMesh
from train_system.runtime import RuntimeContext
from train_system.runtime.plugins.tp import TpPlugin
from train_system.runtime.plugins.sp import SpPlugin


_MODEL_KWARGS = dict(
    dim=64,
    n_heads=4,
    n_kv_heads=4,
    hidden_size=128,
    eps=1e-5,
    n_layers=2,
    vocab_size=256,
    max_seq_len=64,
)

_ATOL = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--use-sp", type=bool, default=False)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29502)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _baseline_loss(model, tokens):
    """Single process baseline loss."""
    model.eval()
    with torch.no_grad():
        loss = model((tokens, tokens.clone()))
    return loss.item()


def _run_worker(rank: int, args: argparse.Namespace, shared_state: dict) -> None:
    group = dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )

    torch.manual_seed(args.seed)
    tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (args.batch_size, args.seq_len))

    if args.use_sp:
        sharded_model = TinyTransformerTpSp(**_MODEL_KWARGS)
    else:
        sharded_model = TinyTransformerTp(**_MODEL_KWARGS)
    # Load the same weights as baseline
    sharded_model.load_state_dict(shared_state["state_dict"])

    mesh = ProcessMesh(dp=1, tp=args.tp_size, pp=1, cp=1, ep=1)
    plan = ParallelPlan(mesh=mesh)
    plugins = [TpPlugin(group)]
    if args.use_sp:
        plugins += [SpPlugin(group)]
    ctx = RuntimeContext(plan=plan, plugins=plugins)
    
    trainer = Trainer(context=ctx, model=sharded_model, optimizer=None)
    trainer.setup()

    trainer.model.eval()
    with torch.no_grad():
        sharded_loss = trainer.model((tokens, tokens.clone()))

    # rank 0 compares
    if rank == 0:
        baseline_loss = shared_state["baseline_loss"]
        sharded_loss = sharded_loss.item()
        print(f"Baseline loss : {baseline_loss:.6f}")
        print(f"sharded loss    : {sharded_loss:.6f}")
        diff = abs(baseline_loss - sharded_loss)
        print(f"Diff          : {diff:.2e}  (atol={_ATOL:.2e})")
        if diff < _ATOL:
            print("PASS ✓")
        else:
            print("FAIL ✗")

    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    assert args.tp_size <= args.world_size

    torch.manual_seed(args.seed)
    tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (args.batch_size, args.seq_len))

    # Baseline: single process, no parallelism
    baseline_model = TinyTransformer(**_MODEL_KWARGS)
    baseline_loss = _baseline_loss(baseline_model, tokens)

    # Share weights and baseline loss with workers via mp shared dict
    shared_state = {
        "state_dict": baseline_model.state_dict(),
        "baseline_loss": baseline_loss,
    }

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args, shared_state), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()