"""Every rank must reach the token-weighted all-reduce, including one with no gradients.

The all-reduce that totals a step's supervised tokens once sat after an early
return taken when a rank held no gradients. A single rank taking that branch
left the others blocked on a one-element ALLREDUCE until the NCCL watchdog
fired ten minutes later and killed a 13B run at step 100.

The process group is given a short timeout so a mismatch raises here instead of
hanging the test the way it hung the run.

  PYTHONPATH=.:tests .venv/bin/python tests/token_weighted_collective_symmetry.py
"""

from __future__ import annotations

from datetime import timedelta
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from models import TinyTransformer
from parallel import ParallelPlan
from runtime import MeshConfig, RuntimeCore

_KW = dict(dim=32, n_heads=4, n_kv_heads=4, hidden_size=64, eps=1e-5,
           n_layers=1, vocab_size=64, max_seq_len=16)


def _worker(rank: int, world: int) -> None:
    dist.init_process_group(
        "gloo", init_method="tcp://127.0.0.1:29637", rank=rank, world_size=world,
        timeout=timedelta(seconds=20),
    )
    core = RuntimeCore(
        mesh=MeshConfig(dp=world, tp=1, pp=1, cp=1, ep=1), plan=ParallelPlan(),
        model=TinyTransformer(**_KW), grad_accum_steps=1,
        token_weighted_loss=True, nominal_tokens_per_microbatch=16,
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.0),
    )
    core.setup()
    core.state.step_context.add_label_tokens(0 if rank == 0 else 8)

    # rank 0 arrives with no gradients at all; every other rank has them.
    if rank != 0:
        for p in core.model.parameters():
            p.grad = torch.zeros_like(p)

    core._finalize_token_weighted_grads()
    dist.barrier()
    if rank == 0:
        print("all ranks reached and left the collective (rank 0 had no gradients)")
    dist.destroy_process_group()


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_worker, args=(2,), nprocs=2, join=True)
    print("PASS")


if __name__ == "__main__":
    main()
