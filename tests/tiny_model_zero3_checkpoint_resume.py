"""Resume test for RuntimeCore ZeRO-3 parameter and optimizer checkpointing.

The test compares:
  1. train step 1, save, train step 2
  2. train step 1, save, rebuild runtime, load, train step 2

AdamW is used so missing optimizer state shows up in the second update.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from models import TinyModel
from parallel import ParallelPlan
from runtime import MeshConfig, RuntimeCore
from runtime.plugins.zero3 import Zero3Plugin
from state import load_sharded_checkpoint, save_sharded_checkpoint


_ATOL = 1e-6
_LR = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29532)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def _build_model(seed: int, hidden_size: int) -> TinyModel:
    torch.manual_seed(seed)
    return TinyModel(hidden_size=hidden_size)


def _build_core(model: TinyModel, world_size: int) -> tuple[RuntimeCore, Zero3Plugin]:
    zero3 = Zero3Plugin(enable_prefetch=True)
    core = RuntimeCore(
        mesh=MeshConfig(dp=world_size, tp=1, pp=1, cp=1, ep=1),
        plan=ParallelPlan(zero_stage=3),
        model=model,
        optimizer_factory=lambda params: torch.optim.AdamW(params, lr=_LR, weight_decay=0.0),
        plugins=[zero3],
    )
    core.setup()
    return core, zero3


def _local_batch(full_batch: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    local_batch_size = full_batch.size(0) // world_size
    return full_batch.narrow(0, rank * local_batch_size, local_batch_size).contiguous()


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

    torch.manual_seed(args.seed)
    first_batch = torch.randn(args.global_batch_size, args.hidden_size)
    second_batch = torch.randn(args.global_batch_size, args.hidden_size)
    first_local_batch = _local_batch(first_batch, rank, args.world_size)
    second_local_batch = _local_batch(second_batch, rank, args.world_size)

    continuous_model = _build_model(args.seed, args.hidden_size)
    continuous_core, continuous_zero3 = _build_core(continuous_model, args.world_size)
    continuous_core.run_train_step(first_local_batch)
    save_sharded_checkpoint(continuous_core.state_manager, args.checkpoint_dir)
    continuous_core.run_train_step(second_local_batch)

    restored_model = _build_model(args.seed, args.hidden_size)
    restored_core, restored_zero3 = _build_core(restored_model, args.world_size)
    load_sharded_checkpoint(restored_core.state_manager, args.checkpoint_dir)
    restored_core.run_train_step(second_local_batch)

    continuous_zero3.materialize_model()
    restored_zero3.materialize_model()
    param_name, param_diff = _max_param_diff(continuous_core.model, restored_core.model)

    if rank == 0:
        print(f"Checkpoint dir    : {args.checkpoint_dir}")
        print(f"Resume diff       : {param_diff:.2e}  ({param_name}, atol={_ATOL:.2e})")
        if param_diff > _ATOL:
            raise AssertionError(f"ZeRO3 checkpoint resume failed: param={param_name}, diff={param_diff:.2e}")
        print("PASS")

    continuous_zero3.reshard_model()
    restored_zero3.reshard_model()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = tempfile.mkdtemp(prefix="zero3_resume_ckpt_")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
