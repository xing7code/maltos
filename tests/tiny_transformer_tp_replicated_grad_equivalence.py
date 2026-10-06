"""Equivalence test: gradients of TP-*replicated* parameters under TP+SP.

Companion to tiny_transformer_tp_grad_clip_equivalence.py, which deliberately
compares only the parameters listed in ``shard_rules`` -- i.e. the TP-sharded
weights.  That filter hides an entire class of parameter: under sequence
parallelism the norm / embedding weights are replicated on every TP rank but
act on a *sequence shard* of the activations, so each rank accumulates only a
partial sum over its own tokens.  The full gradient is the SUM across the TP
dimension (Megatron does this in allreduce_sequence_parallel_grads).

This test compares, for every parameter NOT in shard_rules:

  local    this rank's gradient as-is
  tp_sum   the gradient summed across the TP group
  tp_mean  the gradient averaged across the TP group

against a single-process baseline.  The pattern identifies the bug precisely:

  tp_sum matches, local does not  -> a TP-dimension SUM is missing
  local matches                   -> no bug; replicas are already consistent

It also checks replica consistency directly: if two TP ranks hold different
gradients for the same replicated parameter, their weights diverge on step 1.

Usage:
  PYTHONPATH=. .venv/bin/python tests/tiny_transformer_tp_replicated_grad_equivalence.py
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distributed_test_utils import all_gather_tensor, rule_by_param_name
from helpers import causal_lm_batch
from models import TinyTransformer, TinyTransformerTpSp
from parallel import ParallelPlan
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.plugins.tp_sp import TpSpPlugin


_MODEL_KWARGS = dict(
    dim=64, n_heads=4, n_kv_heads=4, hidden_size=128, eps=1e-5,
    n_layers=2, vocab_size=256, max_seq_len=64,
)
_ATOL = 1e-5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--tp-size", type=int, default=2)
    p.add_argument("--master-addr", type=str, default="127.0.0.1")
    p.add_argument("--master-port", type=int, default=29561)
    p.add_argument("--backend", type=str, default="gloo")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )

    torch.manual_seed(args.seed)
    tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (args.batch_size, args.seq_len))
    baseline = TinyTransformer(**_MODEL_KWARGS)

    baseline.train()
    baseline.zero_grad(set_to_none=True)
    baseline(causal_lm_batch(tokens)).backward()
    baseline_grads = {
        n: (torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
        for n, p in baseline.named_parameters()
    }

    sharded = TinyTransformerTpSp(**_MODEL_KWARGS)
    sharded.load_state_dict(baseline.state_dict())
    core = RuntimeCore(
        mesh=MeshConfig(dp=1, tp=args.tp_size, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=sharded,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.0),
        plugins=[TpSpPlugin(native_comm_overlap=True)],
    )
    core.setup()
    core.model.train()
    core.run_step(causal_lm_batch(tokens))

    tp_group = core.get_group(MeshAxis.TP)
    shard_rules = rule_by_param_name(core.model)

    rows, inconsistent, needs_sum, already_ok = [], [], [], []
    for name, param in core.model.named_parameters():
        if name in shard_rules or name not in baseline_grads:
            continue
        local = torch.zeros_like(param) if param.grad is None else param.grad.detach().clone()
        replicas = all_gather_tensor(local, tp_group)
        spread = max((r - replicas[0]).abs().max().item() for r in replicas)
        tp_sum = torch.stack(replicas).sum(0)
        tp_mean = torch.stack(replicas).mean(0)
        ref = baseline_grads[name]
        d_local = (local - ref).abs().max().item()
        d_sum = (tp_sum - ref).abs().max().item()
        d_mean = (tp_mean - ref).abs().max().item()
        scale = ref.abs().max().item() + 1e-12
        rows.append((name, spread, d_local, d_sum, d_mean, d_local/scale))
        if spread > _ATOL:
            inconsistent.append(name)
        if d_local > _ATOL and d_sum <= _ATOL:
            needs_sum.append(name)
        elif d_local <= _ATOL:
            already_ok.append(name)

    if rank == 0:
        print(f"\nTP={args.tp_size} gloo | replicated parameters (excluded by the existing test)\n")
        print(f"{'parameter':<30}{'replica spread':>15}{'local vs ref':>14}{'rel err':>10}{'TP-sum vs ref':>15}{'TP-mean':>12}")
        print("-" * 96)
        for n, sp, dl, ds, dm, rel in rows:
            print(f"{n:<30}{sp:>15.3e}{dl:>14.3e}{rel:>9.1%}{ds:>15.3e}{dm:>12.3e}")
        print()
        print(f"replicated params checked        : {len(rows)}")
        print(f"gradients differ ACROSS TP ranks : {len(inconsistent)}  {inconsistent}")
        print(f"correct only after a TP SUM      : {len(needs_sum)}  {needs_sum}")
        print(f"already correct rank-local       : {len(already_ok)}")
        print()
        if needs_sum:
            print("VERDICT: confirmed -- a TP-dimension SUM is missing for these parameters.")
        elif inconsistent:
            print("VERDICT: replicas disagree, but a plain TP SUM does not recover the reference.")
        else:
            print("VERDICT: no gap found -- replicated gradients already match the single-process reference.")

    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    assert args.tp_size == args.world_size
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
