"""Regression: ZeRO-3 must place static model buffers with its local shards."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from models import OlmoConfig, OlmoForCausalLM, OlmoRMSNorm
from parallel import ParallelPlan
from runtime import MeshConfig, RuntimeCore
from runtime.layers.distributed_rmsnorm import DistributedRMSNorm
from runtime.plugins.zero3 import Zero3Plugin


def _worker(rank: int, port: int) -> None:
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        model = OlmoForCausalLM(
            OlmoConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                max_position_embeddings=8,
            )
        )
        core = RuntimeCore(
            mesh=MeshConfig(dp=2),
            plan=ParallelPlan(),
            model=model,
            device="cpu",
            dtype=torch.bfloat16,
            optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.01),
            plugins=[Zero3Plugin({nn.Linear, nn.Embedding, OlmoRMSNorm, DistributedRMSNorm})],
        )
        core.setup()
        try:
            for rotary in [model.rotary_emb, *(layer.self_attn.rotary_emb for layer in model.layers)]:
                assert rotary.cos.device.type == "cpu"
                assert rotary.sin.device.type == "cpu"
                assert rotary.cos.dtype == torch.bfloat16
                assert rotary.sin.dtype == torch.bfloat16
        finally:
            core.close()
    finally:
        dist.destroy_process_group()


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_worker, args=(29631,), nprocs=2, join=True)
    print("zero3 buffer materialization: PASS")


if __name__ == "__main__":
    main()
