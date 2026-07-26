"""One-step RuntimeCore contract for FCP-style HDP-balanced CP.

Usage:
  PYTHONPATH=. .venv/bin/python tests/hdp_runtime_core_equivalence.py
"""

from __future__ import annotations

import copy
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from models.tiny_transformer import RmsNorm, TinyTransformer
from parallel.plan import ParallelPlan
from runtime.core import RuntimeCore
from runtime.mesh import MeshConfig
from runtime.plugins.ddp import BucketDataParallelPlugin, DataParallelPlugin
from data.bytescale_hdp import build_bytescale_local_batches
from parallel.hdp_helper import ByteScaleHdpBalancedConfig
from parallel.hdp_helper import BYTESCALE_HDP_WAVES_KEY
from runtime.plugins.hdp import ByteScaleHdpPlugin
from runtime.plugins.zero1 import Zero1Plugin
from runtime.plugins.zero2 import Zero2Plugin
from runtime.plugins.zero3 import Zero3Plugin
from utils.constants import INPUT_IDS_KEY, LABELS_KEY, POSITION_IDS_KEY, SEQUENCE_IDS_KEY


_PORT = 29683
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


def _plugins(mode: str):
    hdp = ByteScaleHdpPlugin(config=ByteScaleHdpBalancedConfig(partition_tokens=6))
    if mode == "naive_ddp":
        return [DataParallelPlugin(), hdp]
    if mode == "bucket_ddp":
        return [BucketDataParallelPlugin(bucket_mb_size=1), hdp]
    if mode == "zero1":
        return [hdp, Zero1Plugin()]
    if mode == "zero2":
        return [hdp, Zero2Plugin()]
    if mode == "zero3":
        return [hdp, Zero3Plugin(wrap_cls={nn.Linear, nn.Embedding, RmsNorm})]
    raise ValueError(f"unknown mode={mode}")


def _run_worker(rank: int, mode: str) -> None:
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{_PORT}", rank=rank, world_size=2)
    try:
        torch.manual_seed(41)
        reference_model = _model()
        hdp_model = copy.deepcopy(reference_model)
        # One 8-token document needs both HDP ranks; the packed 2-token
        # document is assigned only to rank 0.  This exercises unequal local
        # work plus the token-weighted loss correction before DDP averaging.
        batch = {
            INPUT_IDS_KEY: torch.randint(0, 32, (1, 10)),
            LABELS_KEY: torch.randint(0, 32, (1, 10)),
            POSITION_IDS_KEY: torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1]]),
            SEQUENCE_IDS_KEY: torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]),
        }

        reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=1e-2)
        reference_loss = reference_model(batch)
        reference_loss.backward()
        reference_optimizer.step()

        runtime = RuntimeCore(
            model=hdp_model,
            mesh=MeshConfig(dp=2, tp=1, pp=1, cp=1, ep=1),
            plan=ParallelPlan(),
            optimizer_factory=lambda params: torch.optim.SGD(params, lr=1e-2),
            plugins=_plugins(mode),
        )
        runtime.setup()
        local_waves, _ = build_bytescale_local_batches(
            batch,
            rank=rank,
            world_size=2,
            config=ByteScaleHdpBalancedConfig(partition_tokens=6),
        )
        local_loss, should_step = runtime.run_step({BYTESCALE_HDP_WAVES_KEY: local_waves})
        assert should_step
        runtime.step_optimizer()

        loss = local_loss.detach().clone()
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        worst = (loss - reference_loss.detach()).abs()
        for plugin in runtime.plugins:
            materialize = getattr(plugin, "materialize_model", None)
            if callable(materialize):
                materialize()
        for (_, expected), (_, actual) in zip(reference_model.named_parameters(), runtime.model.named_parameters(), strict=True):
            worst = torch.maximum(worst, (expected - actual).abs().max())
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"HDP {mode} max diff: {worst.item():.2e} (atol={_ATOL:.2e})")
            if worst.item() > _ATOL:
                raise AssertionError(f"HDP RuntimeCore mismatch: {worst.item():.2e}")
            print("PASS")
        runtime.close()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for mode in ("naive_ddp", "bucket_ddp", "zero1", "zero2", "zero3"):
        mp.spawn(_run_worker, args=(mode,), nprocs=2, join=True)


if __name__ == "__main__":
    main()
