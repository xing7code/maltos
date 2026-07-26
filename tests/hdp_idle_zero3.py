"""Regression: an idle HDP rank must still execute ZeRO-3 module hooks.

Usage:
  PYTHONPATH=. .venv/bin/python tests/hdp_idle_zero3.py
"""
from __future__ import annotations

import copy
from datetime import timedelta
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from data.bytescale_hdp import build_bytescale_local_batch
from models.tiny_transformer import RmsNorm, TinyTransformer
from parallel.plan import ParallelPlan
from parallel.hdp_helper import ByteScaleHdpBalancedConfig
from parallel.hdp_helper import BYTESCALE_HDP_WAVES_KEY
from runtime.core import RuntimeCore
from runtime.mesh import MeshConfig
from runtime.plugins.hdp import ByteScaleHdpPlugin
from runtime.plugins.zero3 import Zero3Plugin
from utils.constants import INPUT_IDS_KEY, LABELS_KEY, POSITION_IDS_KEY, SEQUENCE_IDS_KEY


_PORT = 29684
_ATOL = 3e-5


def _model() -> TinyTransformer:
    return TinyTransformer(
        dim=8,
        n_heads=2,
        n_kv_heads=2,
        hidden_size=16,
        eps=1e-5,
        n_layers=1,
        vocab_size=32,
        max_seq_len=8,
        attention_backend="eager",
    )


def _run_worker(rank: int) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{_PORT}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(91)
        reference_model = _model()
        hdp_model = copy.deepcopy(reference_model)
        batch = {
            INPUT_IDS_KEY: torch.randint(0, 32, (1, 4)),
            LABELS_KEY: torch.randint(0, 32, (1, 4)),
            POSITION_IDS_KEY: torch.arange(4).unsqueeze(0),
            SEQUENCE_IDS_KEY: torch.zeros((1, 4), dtype=torch.long),
        }

        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=1e-2)
        reference_loss = reference_model(batch)
        reference_loss.backward()
        reference_optimizer.step()

        zero3 = Zero3Plugin(wrap_cls={nn.Linear, nn.Embedding, RmsNorm})
        runtime = RuntimeCore(
            model=hdp_model,
            mesh=MeshConfig(dp=2, tp=1, pp=1, cp=1, ep=1),
            plan=ParallelPlan(),
            optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=1e-2),
            plugins=[
                ByteScaleHdpPlugin(config=ByteScaleHdpBalancedConfig(partition_tokens=4)),
                zero3,
            ],
        )
        runtime.setup()
        local_batch, layout = build_bytescale_local_batch(
            batch,
            rank=rank,
            world_size=2,
            config=ByteScaleHdpBalancedConfig(partition_tokens=4),
        )
        assert bool(layout.documents) is (rank == 0)
        loss, should_step = runtime.run_step({BYTESCALE_HDP_WAVES_KEY: (local_batch,)})
        assert should_step
        runtime.step_optimizer()

        average_loss = loss.detach().clone()
        dist.all_reduce(average_loss, op=dist.ReduceOp.AVG)
        zero3.materialize_model()
        worst = (average_loss - reference_loss.detach()).abs()
        for (_, expected), (_, actual) in zip(
            reference_model.named_parameters(),
            runtime.model.named_parameters(),
            strict=True,
        ):
            worst = torch.maximum(worst, (expected - actual).abs().max())
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"HDP idle ZeRO-3 max diff: {worst.item():.2e} (atol={_ATOL:.2e})")
            if worst.item() > _ATOL:
                raise AssertionError(f"HDP idle ZeRO-3 mismatch: {worst.item():.2e}")
            print("PASS")
        zero3.reshard_model()
        runtime.close()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, nprocs=2, join=True)


if __name__ == "__main__":
    main()
