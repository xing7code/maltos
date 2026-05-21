"""Equivalence test: full-batch TinyModel vs RuntimeCore DDP v2.

Each rank sees a different local batch. DataParallelPlugin averages gradients
across the DP group, which should match a single-process full-batch baseline.

Usage:
  PYTHONPATH=. .venv/bin/python train_system/tests/tiny_model_ddp_runtime_core_equivalence.py \
    --world-size 2 \
    --ddp-mode naive
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
from train_system.runtime.plugins.ddp import BucketDataParallelPlugin, DataParallelPlugin


_ATOL = 1e-6
_LR = 1e-2


class _NoOpOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {})

    def step(self, closure=None):
        if closure is not None:
            return closure()
        return None

    def zero_grad(self, set_to_none: bool = True):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29517)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ddp-mode", choices=("naive", "async", "bucket"), default="naive")
    parser.add_argument("--bucket-mb-size", type=int, default=25)
    return parser.parse_args()


def _build_reference(seed: int, global_batch_size: int, hidden_size: int) -> tuple[TinyModel, torch.Tensor]:
    torch.manual_seed(seed)
    batch = torch.randn(global_batch_size, hidden_size)
    model = TinyModel(hidden_size=hidden_size)
    return model, batch


def _max_grad_diff(lhs: TinyModel, rhs: TinyModel) -> tuple[str, float]:
    worst_name = ""
    worst_diff = 0.0
    for (lhs_name, lhs_param), (rhs_name, rhs_param) in zip(lhs.named_parameters(), rhs.named_parameters()):
        assert lhs_name == rhs_name
        assert lhs_param.grad is not None
        assert rhs_param.grad is not None
        diff = (lhs_param.grad - rhs_param.grad).abs().max().item()
        if diff > worst_diff:
            worst_name = lhs_name
            worst_diff = diff
    return worst_name, worst_diff


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
    ddp_model = TinyModel(hidden_size=args.hidden_size)
    ddp_model.load_state_dict(baseline_model.state_dict())

    local_batch_size = args.global_batch_size // args.world_size
    local_batch = full_batch.narrow(0, rank * local_batch_size, local_batch_size).contiguous()

    baseline_optimizer = torch.optim.SGD(baseline_model.parameters(), lr=_LR)
    ddp_optimizer = torch.optim.SGD(ddp_model.parameters(), lr=_LR)
    baseline_optimizer.zero_grad(set_to_none=True)
    ddp_optimizer.zero_grad(set_to_none=True)

    baseline_loss = baseline_model(full_batch)
    baseline_loss.backward()

    if args.ddp_mode == "naive":
        plugins = [DataParallelPlugin(async_op=False)]
    elif args.ddp_mode == "async":
        plugins = [DataParallelPlugin(async_op=True)]
    elif args.ddp_mode == "bucket":
        plugins = [BucketDataParallelPlugin(bucket_mb_size=args.bucket_mb_size)]
    else:
        raise ValueError(f"unknown ddp_mode={args.ddp_mode}")

    core = RuntimeCore(
        mesh=MeshConfig(dp=args.world_size, tp=1, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=ddp_model,
        optimizer=_NoOpOptimizer(ddp_model.parameters()),
        plugins=plugins,
    )
    core.setup()
    local_loss = core.run_train_step(local_batch)

    avg_loss = local_loss.detach().clone()
    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)

    grad_name, grad_diff = _max_grad_diff(baseline_model, core.model)
    baseline_optimizer.step()
    ddp_optimizer.step()
    param_name, param_diff = _max_param_diff(baseline_model, core.model)

    if rank == 0:
        loss_diff = abs(baseline_loss.item() - avg_loss.item())
        print(f"Baseline loss     : {baseline_loss.item():.6f}")
        print(f"RuntimeCore DDP   : {avg_loss.item():.6f}  ({args.ddp_mode})")
        print(f"Loss diff         : {loss_diff:.2e}  (atol={_ATOL:.2e})")
        print(f"Grad diff         : {grad_diff:.2e}  ({grad_name}, atol={_ATOL:.2e})")
        print(f"Post-step diff    : {param_diff:.2e}  ({param_name}, atol={_ATOL:.2e})")
        if loss_diff > _ATOL:
            raise AssertionError(f"DDP loss equivalence failed: diff={loss_diff:.2e}")
        if grad_diff > _ATOL:
            raise AssertionError(f"DDP gradient equivalence failed: param={grad_name}, diff={grad_diff:.2e}")
        if param_diff > _ATOL:
            raise AssertionError(f"DDP one-step equivalence failed: param={param_name}, diff={param_diff:.2e}")
        print("PASS")

    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
