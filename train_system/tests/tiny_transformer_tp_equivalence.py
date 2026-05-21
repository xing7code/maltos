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


def _str_to_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in ("1", "true", "t", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--use-sp", type=_str_to_bool, default=False)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29502)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _build_reference(seed: int, batch_size: int, seq_len: int) -> tuple[TinyTransformer, torch.Tensor]:
    torch.manual_seed(seed)
    tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len))
    model = TinyTransformer(**_MODEL_KWARGS)
    return model, tokens


def _baseline_loss(model: TinyTransformer, tokens: torch.Tensor) -> float:
    """Single process baseline loss."""
    model.eval()
    with torch.no_grad():
        loss = model((tokens, tokens.clone()))
    return loss.item()


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    group = dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )

    baseline_model, tokens = _build_reference(args.seed, args.batch_size, args.seq_len)
    baseline_loss = _baseline_loss(baseline_model, tokens)

    if args.use_sp:
        sharded_model = TinyTransformerTpSp(**_MODEL_KWARGS)
    else:
        sharded_model = TinyTransformerTp(**_MODEL_KWARGS)
    # Load the same weights as baseline
    sharded_model.load_state_dict(baseline_model.state_dict())

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
        sharded_loss = sharded_loss.item()
        print(f"Baseline loss : {baseline_loss:.6f}")
        print(f"sharded loss    : {sharded_loss:.6f}")
        diff = abs(baseline_loss - sharded_loss)
        print(f"Diff          : {diff:.2e}  (atol={_ATOL:.2e})")
        if diff <= _ATOL:
            print("PASS ✓")
        else:
            print("FAIL ✗")
            raise AssertionError(
                f"TP equivalence failed: baseline_loss={baseline_loss:.6f}, "
                f"sharded_loss={sharded_loss:.6f}, diff={diff:.2e}, atol={_ATOL:.2e}"
            )

    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    assert args.tp_size <= args.world_size

    torch.manual_seed(args.seed)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
