"""Smoke test for TinyTransformer with the Trainer loop via mp.spawn.

Single process:
  PYTHONPATH=. .venv/bin/python train_system/tests/smoke_tiny_transformer.py --world-size 1

Multi-process (CPU/gloo):
  PYTHONPATH=. .venv/bin/python train_system/tests/smoke_tiny_transformer.py --world-size 2

TP:
  PYTHONPATH=. .venv/bin/python train_system/tests/smoke_tiny_transformer.py --world-size 2 --tp-size 2 --model-class tiny_tp --v 3

TP+SP:
  PYTHONPATH=. .venv/bin/python tools/smoke_tiny_transformer.py --world-size 2 --tp-size 2 --model-class tiny_tp_sp --use-sp true --v 3
"""

from __future__ import annotations

import argparse
import os
from absl import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train_system.engine import Trainer
from train_system.examples import TinyTransformer, TinyTransformerTp, TinyTransformerTpSp
from train_system.parallel import ParallelConfig, ParallelPlan, ProcessMesh
from train_system.runtime import RuntimeContext
from train_system.runtime.plugins.tp import TpPlugin
from train_system.runtime.plugins.sp import SpPlugin


_REGISTRY_MODLE_CLASS = {
    "tiny": TinyTransformer,
    "tiny_tp": TinyTransformerTp,
    "tiny_tp_sp": TinyTransformerTpSp,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyTransformer smoke test with mp.spawn")
    parser.add_argument("--world-size", type=int, default=2, help="Number of local processes")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1", help="Master address for c10d init")
    parser.add_argument("--master-port", type=int, default=29501, help="Master port for c10d init")
    parser.add_argument("--backend", type=str, default="gloo", help="Process group backend")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor Parallel size to test.")
    parser.add_argument("--use-sp", type=bool, default=False, help="Whether to use sp along with tp.")
    parser.add_argument("--model-class", type=str, default="tiny", help="The model class to test on.")
    parser.add_argument("--steps", type=int, default=3, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size")
    parser.add_argument("-v", "--v", type=int, default=0, help="Verbosity level for VLOG")
    return parser.parse_args()


def _data_iter(vocab_size: int, batch_size: int, seq_len: int):
    while True:
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        yield (tokens, tokens.clone())


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    logging.set_verbosity(args.v)
    logging.use_absl_handler()
    world_size = args.world_size
    use_dist = world_size > 1
    use_tp = args.tp_size > 1
    group = dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=world_size,
    )
    plan = ParallelPlan(
        mesh=ProcessMesh(dp=world_size, tp=args.tp_size, pp=1, cp=1, ep=1),
        config=ParallelConfig(use_ddp=use_dist, use_tp=use_tp, use_sp=args.use_sp, use_pp=False, zero_stage=0),
    )
    plugins = []
    if use_tp:
        plugins += [TpPlugin(group)]
    if use_tp and args.use_sp:
        plugins += [SpPlugin(group)]
        
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
    )
    trainer = Trainer(context=ctx, model=model, optimizer=None)
    trainer.setup()
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)

    trainer.train_steps(
        _data_iter(vocab_size=args.vocab_size, batch_size=args.batch_size, seq_len=args.seq_len),
        steps=args.steps,
    )
    logging.info(f"[rank={rank}] tiny transformer smoke ok")

    if use_dist:
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    assert args.tp_size <= args.world_size, "TP maximally can only up to world_size."
    assert args.model_class in _REGISTRY_MODLE_CLASS, f"invalid model_class {args.model_class}"
    if args.world_size == 1:
        _run_worker(rank=0, args=args)
        return

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
