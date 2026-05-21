"""Equivalence test: baseline TinyTransformer vs RuntimeCore TP v2.

This is the migration gate for moving TP from the legacy Trainer/BasePlugin
path to RuntimeCore/RuntimePlugin.

Usage:
  PYTHONPATH=. .venv/bin/python train_system/tests/tiny_transformer_tp_runtime_core_equivalence.py \
    --world-size 2 \
    --tp-size 2
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train_system.examples import TinyTransformer, TinyTransformerTp
from train_system.parallel import ParallelPlan
from train_system.runtime import MeshConfig, RuntimeCore
from train_system.runtime.plugins.tp_v2 import TensorParallelPluginV2


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
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29515)
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
    model.eval()
    with torch.no_grad():
        loss = model((tokens, tokens.clone()))
    return loss.item()


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )

    baseline_model, tokens = _build_reference(args.seed, args.batch_size, args.seq_len)
    baseline_loss = _baseline_loss(baseline_model, tokens)

    sharded_model = TinyTransformerTp(**_MODEL_KWARGS)
    sharded_model.load_state_dict(baseline_model.state_dict())

    core = RuntimeCore(
        mesh=MeshConfig(dp=1, tp=args.tp_size, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=sharded_model,
        optimizer=None,
        plugins=[TensorParallelPluginV2()],
    )
    core.setup()
    core.model.eval()

    with torch.no_grad():
        sharded_loss = core.model((tokens, tokens.clone()))

    if rank == 0:
        sharded_loss = sharded_loss.item()
        diff = abs(baseline_loss - sharded_loss)
        print(f"Baseline loss : {baseline_loss:.6f}")
        print(f"RuntimeCore TP: {sharded_loss:.6f}")
        print(f"Diff          : {diff:.2e}  (atol={_ATOL:.2e})")
        if diff <= _ATOL:
            print("PASS")
        else:
            raise AssertionError(
                f"RuntimeCore TP equivalence failed: baseline_loss={baseline_loss:.6f}, "
                f"sharded_loss={sharded_loss:.6f}, diff={diff:.2e}, atol={_ATOL:.2e}"
            )

    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    assert args.tp_size == args.world_size

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
