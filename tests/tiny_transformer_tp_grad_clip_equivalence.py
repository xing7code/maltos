"""Equivalence test: global gradient norm reported by GradClipPlugin under TP.

GradClipPlugin must all-reduce the squared norms across all TP ranks to report
the *global* gradient norm, not each rank's local parameter-shard norm.

Before the fix the plugin called torch.nn.utils.clip_grad_norm_ directly on
each rank's local parameters, which reports ≈ 70 % of the true global norm
(a 29 % relative error for TP=2).  After the fix an all_reduce gives the
correct global norm; the residual error (< 0.5 %) is from RMSNorm parameters
being replicated across TP ranks and therefore double-counted.

Strategy: set max_norm very large so no clipping fires, then read
metadata["grad_norm"] from RuntimeCore.state and compare against the norm
computed from a single-rank TinyTransformer baseline.

Usage:
  PYTHONPATH=. .venv/bin/python tests/tiny_transformer_tp_grad_clip_equivalence.py
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distributed_test_utils import rule_by_param_name as _rule_by_param_name
from helpers import causal_lm_batch
from models import TinyTransformer, TinyTransformerTpSp
from parallel import ParallelPlan
from runtime import MeshConfig, RuntimeCore
from runtime.plugins.grad_clip import GradClipPlugin
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

_LR = 1e-2
_NORM_REL_TOL = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29547)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _build_reference(
    seed: int, batch_size: int, seq_len: int
) -> tuple[TinyTransformer, torch.Tensor]:
    torch.manual_seed(seed)
    tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len))
    model = TinyTransformer(**_MODEL_KWARGS)
    return model, tokens


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )

    # All ranks build the same baseline model and tokens (same seed).
    baseline_model, tokens = _build_reference(args.seed, args.batch_size, args.seq_len)

    # Baseline: single-rank full forward+backward to compute reference global norm.
    baseline_optimizer = torch.optim.SGD(baseline_model.parameters(), lr=_LR)
    baseline_optimizer.zero_grad(set_to_none=True)
    baseline_model.train()
    baseline_loss = baseline_model(causal_lm_batch(tokens))
    baseline_loss.backward()
    baseline_norm = torch.nn.utils.clip_grad_norm_(
        baseline_model.parameters(), max_norm=float("inf")
    ).item()

    # RuntimeCore TP+SP: max_norm is set large enough that no clipping fires.
    # GradClipPlugin still computes and records the global norm in metadata.
    sharded_model = TinyTransformerTpSp(**_MODEL_KWARGS)
    sharded_model.load_state_dict(baseline_model.state_dict())
    core = RuntimeCore(
        mesh=MeshConfig(dp=1, tp=args.tp_size, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=sharded_model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=[
            TensorParallelPlugin(),
            SequenceParallelPlugin(),
            GradClipPlugin(max_norm=1e10),
        ],
    )
    core.setup()
    core.model.train()
    _, _ = core.run_step(causal_lm_batch(tokens))
    core.step_optimizer()

    reported_norm = core.state.metadata.get("grad_norm")

    if rank == 0:
        if reported_norm is None:
            raise AssertionError("GradClipPlugin did not set metadata['grad_norm']")
        rel_err = abs(reported_norm - baseline_norm) / (baseline_norm + 1e-12)
        print(f"Baseline grad norm  : {baseline_norm:.6f}")
        print(f"Reported grad norm  : {reported_norm:.6f}")
        print(f"Relative error      : {rel_err:.4f}  (tol={_NORM_REL_TOL:.2f})")
        if rel_err > _NORM_REL_TOL:
            raise AssertionError(
                f"TP+GradClip norm mismatch: baseline={baseline_norm:.6f}, "
                f"reported={reported_norm:.6f}, rel_err={rel_err:.4f}"
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
