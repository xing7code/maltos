"""Equivalence test: baseline TinyTransformer vs RuntimeCore TP/SP v2.

This is the migration gate for TP and SP on the RuntimeCore/RuntimePlugin path.

Usage:
  PYTHONPATH=. .venv/bin/python tests/tiny_transformer_tp_runtime_core_equivalence.py \
    --world-size 2 \
    --tp-size 2

  PYTHONPATH=. .venv/bin/python tests/tiny_transformer_tp_runtime_core_equivalence.py \
    --world-size 2 \
    --tp-size 2 \
    --use-sp true
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distributed_test_utils import (
    max_diff as _max_diff,
    named_tensors as _named_tensors,
    rule_by_param_name as _rule_by_param_name,
)
from models import TinyTransformer, TinyTransformerTp, TinyTransformerTpSp
from helpers import causal_lm_batch
from parallel import ParallelPlan
from runtime import MeshConfig, RuntimeCore
from runtime.plugins.sp import SequenceParallelPlugin
from runtime.plugins.tp import TensorParallelPlugin


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
_GRAD_ATOL = 2e-2
_STEP_ATOL = 5e-4
_LR = 1e-2


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
        loss = model(causal_lm_batch(tokens))
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

    sharded_cls = TinyTransformerTpSp if args.use_sp else TinyTransformerTp
    sharded_model = sharded_cls(**_MODEL_KWARGS)
    sharded_model.load_state_dict(baseline_model.state_dict())

    plugins = [TensorParallelPlugin()]
    if args.use_sp:
        plugins.append(SequenceParallelPlugin())

    core = RuntimeCore(
        mesh=MeshConfig(dp=1, tp=args.tp_size, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=sharded_model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.0),
        plugins=plugins,
    )
    core.setup()
    baseline_model.train()
    core.model.train()

    baseline_optimizer = torch.optim.SGD(baseline_model.parameters(), lr=_LR)
    sharded_optimizer = torch.optim.SGD(core.model.parameters(), lr=_LR)

    baseline_optimizer.zero_grad(set_to_none=True)
    sharded_optimizer.zero_grad(set_to_none=True)

    baseline_train_loss = baseline_model(causal_lm_batch(tokens))
    sharded_train_loss = core.model(causal_lm_batch(tokens))
    baseline_train_loss.backward()
    sharded_train_loss.backward()

    shard_rules = _rule_by_param_name(sharded_model)
    baseline_grads = _named_tensors(baseline_model, {}, None, grads=True)
    sharded_grads = _named_tensors(core.model, shard_rules, dist.group.WORLD, grads=True)
    grad_name, grad_diff = _max_diff(baseline_grads, sharded_grads)

    baseline_optimizer.step()
    sharded_optimizer.step()

    baseline_params = _named_tensors(baseline_model, {}, None)
    sharded_params = _named_tensors(core.model, shard_rules, dist.group.WORLD)
    step_name, step_diff = _max_diff(baseline_params, sharded_params)

    if rank == 0:
        sharded_loss = sharded_train_loss.item()
        diff = abs(baseline_loss - sharded_loss)
        runtime_label = "RuntimeCore TP+SP" if args.use_sp else "RuntimeCore TP"
        print(f"Baseline loss : {baseline_loss:.6f}")
        print(f"{runtime_label}: {sharded_loss:.6f}")
        print(f"Diff          : {diff:.2e}  (atol={_ATOL:.2e})")
        print(f"Grad diff     : {grad_diff:.2e}  ({grad_name}, atol={_GRAD_ATOL:.2e})")
        print(f"Step diff     : {step_diff:.2e}  ({step_name}, atol={_STEP_ATOL:.2e})")
        if diff > _ATOL:
            raise AssertionError(
                f"RuntimeCore TP equivalence failed: baseline_loss={baseline_loss:.6f}, "
                f"sharded_loss={sharded_loss:.6f}, diff={diff:.2e}, atol={_ATOL:.2e}"
            )
        if grad_diff > _GRAD_ATOL:
            raise AssertionError(
                f"RuntimeCore gradient equivalence failed: param={grad_name}, "
                f"diff={grad_diff:.2e}, atol={_GRAD_ATOL:.2e}"
            )
        if step_diff > _STEP_ATOL:
            raise AssertionError(
                f"RuntimeCore one-step equivalence failed: param={step_name}, "
                f"diff={step_diff:.2e}, atol={_STEP_ATOL:.2e}"
            )
        print("PASS")

    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    assert args.tp_size == args.world_size

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
