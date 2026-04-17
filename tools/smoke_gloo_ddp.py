"""Local smoke test for distributed init on Mac/Linux CPU.

Run:
  torchrun --standalone --nproc_per_node=2 tools/smoke_gloo_ddp.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from train_system.engine import Trainer
from train_system.examples import TinyModel
from train_system.parallel import ParallelConfig, ParallelPlan, ProcessMesh
from train_system.runtime import RuntimeContext


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="gloo")

    plan = ParallelPlan(
        mesh=ProcessMesh(dp=world_size, tp=1, pp=1, cp=1, ep=1),
        config=ParallelConfig(use_ddp=True, use_tp=False, use_pp=False, zero_stage=0),
    )

    ctx = RuntimeContext(plan=plan)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(context=ctx, model=model, optimizer=optimizer)
    trainer.setup()

    # Fake local data iterator
    def _data_iter():
        while True:
            yield torch.randn(8, 32)

    trainer.train_steps(_data_iter(), steps=3)
    print(f"[rank={rank}] smoke run ok")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
