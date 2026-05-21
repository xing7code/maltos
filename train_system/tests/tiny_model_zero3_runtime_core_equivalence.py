"""Equivalence test: full-batch TinyModel vs RuntimeCore ZeRO-3 v2.

Each rank sees a different local batch. Zero3PluginV2 keeps wrapped module
parameters sharded, materializes full params around forward/backward compute,
reduce-scatters averaged gradients into local shards, and steps only shard
parameters.

Usage:
  PYTHONPATH=. .venv/bin/python train_system/tests/tiny_model_zero3_runtime_core_equivalence.py \
    --world-size 2
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train_system.examples import TinyModel
from train_system.parallel import ParallelPlan
from train_system.runtime import MeshConfig, RuntimeCore
from train_system.runtime.plugins.zero3_v2 import Zero3PluginV2


_ATOL = 1e-6
_LR = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29522)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--disable-prefetch", action="store_true")
    return parser.parse_args()


def _build_reference(seed: int, global_batch_size: int, hidden_size: int) -> tuple[TinyModel, torch.Tensor]:
    torch.manual_seed(seed)
    batch = torch.randn(global_batch_size, hidden_size)
    model = TinyModel(hidden_size=hidden_size)
    return model, batch


def _max_param_diff(lhs: TinyModel, rhs: TinyModel) -> tuple[str, float]:
    worst_name = ""
    worst_diff = 0.0
    for (lhs_name, lhs_param), (rhs_name, rhs_param) in zip(lhs.named_parameters(), rhs.named_parameters()):
        assert lhs_name == rhs_name
        diff = (lhs_param.detach() - rhs_param.detach()).abs().max().item()
        if diff > worst_diff:
            worst_name = lhs_name
            worst_diff = diff
    return worst_name, worst_diff


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )

    if args.global_batch_size % args.world_size != 0:
        raise ValueError("global batch size must be divisible by world size")

    baseline_model, full_batch = _build_reference(args.seed, args.global_batch_size, args.hidden_size)
    zero_model = TinyModel(hidden_size=args.hidden_size)
    zero_model.load_state_dict(baseline_model.state_dict())

    local_batch_size = args.global_batch_size // args.world_size
    local_batch = full_batch.narrow(0, rank * local_batch_size, local_batch_size).contiguous()

    baseline_optimizer = torch.optim.SGD(baseline_model.parameters(), lr=_LR)
    baseline_optimizer.zero_grad(set_to_none=True)
    baseline_loss = baseline_model(full_batch)
    baseline_loss.backward()

    zero3 = Zero3PluginV2(enable_prefetch=not args.disable_prefetch, optimizer_cls=torch.optim.SGD, lr=_LR)
    core = RuntimeCore(
        mesh=MeshConfig(dp=args.world_size, tp=1, pp=1, cp=1, ep=1),
        plan=ParallelPlan(zero_stage=3),
        model=zero_model,
        optimizer=None,
        plugins=[zero3],
    )
    core.setup()
    local_loss = core.run_train_step(local_batch)
    baseline_optimizer.step()

    avg_loss = local_loss.detach().clone()
    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    zero3.materialize_model()
    param_name, param_diff = _max_param_diff(baseline_model, core.model)

    if rank == 0:
        loss_diff = abs(baseline_loss.item() - avg_loss.item())
        print(f"Baseline loss     : {baseline_loss.item():.6f}")
        mode = "no-prefetch" if args.disable_prefetch else "prefetch"
        print(f"RuntimeCore ZeRO3 : {avg_loss.item():.6f}  ({mode})")
        print(f"Loss diff         : {loss_diff:.2e}  (atol={_ATOL:.2e})")
        print(f"Post-step diff    : {param_diff:.2e}  ({param_name}, atol={_ATOL:.2e})")
        if loss_diff > _ATOL:
            raise AssertionError(f"ZeRO3 loss equivalence failed: diff={loss_diff:.2e}")
        if param_diff > _ATOL:
            raise AssertionError(f"ZeRO3 one-step equivalence failed: param={param_name}, diff={param_diff:.2e}")
        print("PASS")

    zero3.reshard_model()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
