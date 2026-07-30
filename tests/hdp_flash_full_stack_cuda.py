"""CUDA/NCCL HDP full-stack coverage for Flash, GQA, checkpointing and TP/SP.

Run on a four-GPU host:
  PYTHONPATH=. .venv/bin/python tests/hdp_flash_full_stack_cuda.py
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data.bytescale_hdp import build_bytescale_local_batches
from models import ActivationCheckpointConfig, LlamaConfig, LlamaForCausalLM, LlamaForCausalLMTpSp
from parallel.hdp_helper import BYTESCALE_HDP_WAVES_KEY, ByteScaleHdpBalancedConfig
from parallel.plan import ParallelPlan
from runtime.core import RuntimeCore
from runtime.mesh import MeshConfig
from runtime.plugins.hdp import ByteScaleHdpPlugin
from runtime.plugins.tp_sp import TpSpPlugin
from runtime.plugins.zero1 import Zero1Plugin
from utils.constants import INPUT_IDS_KEY, LABELS_KEY, POSITION_IDS_KEY, SEQUENCE_IDS_KEY
from runtime.layers.flash_utils import flash_attn_block_fallback_reason


_PORT = 29685


def _config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=16, attention_backend="flash_attn",
        activation_checkpointing=ActivationCheckpointConfig(enabled=True, every_n_layers=1),
    )


def _batch() -> dict[str, torch.Tensor]:
    # An 8-token document plus a 2-token document forces multiple HDP waves.
    generator = torch.Generator(device="cpu").manual_seed(917)
    return {
        INPUT_IDS_KEY: torch.randint(0, 64, (1, 10), generator=generator),
        LABELS_KEY: torch.randint(0, 64, (1, 10), generator=generator),
        POSITION_IDS_KEY: torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1]]),
        SEQUENCE_IDS_KEY: torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]),
    }


def _to_device(waves, device: torch.device):
    return tuple({key: value.to(device) if torch.is_tensor(value) else value for key, value in wave.items()} for wave in waves)


def _run_case(rank: int, tp_sp: bool) -> None:
    world_size = 4 if tp_sp else 2
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{_PORT + int(tp_sp)}", rank=rank, world_size=world_size)
    try:
        # TP/HDP ranks must begin from identical logical parameters.
        torch.manual_seed(41)
        torch.cuda.manual_seed_all(41)
        tp = 2 if tp_sp else 1
        dp = 2
        hdp_rank = rank // tp
        model_type = LlamaForCausalLMTpSp if tp_sp else LlamaForCausalLM
        model = model_type(_config()).to(device=device, dtype=torch.bfloat16)
        config = ByteScaleHdpBalancedConfig(partition_tokens=6 if not tp_sp else 4)
        plugins = [ByteScaleHdpPlugin(config=config), Zero1Plugin()]
        if tp_sp:
            plugins = [TpSpPlugin(), *plugins]
        runtime = RuntimeCore(
            model=model, mesh=MeshConfig(dp=dp, tp=tp, pp=1, cp=1, ep=1), plan=ParallelPlan(),
            optimizer_factory=lambda params: torch.optim.SGD(params, lr=1e-3), plugins=plugins,
        )
        runtime.setup()
        waves, layouts = build_bytescale_local_batches(_batch(), rank=hdp_rank, world_size=dp, config=config)
        assert len(waves) > 1, "full-stack case must exercise multiple waves"
        assert any(
            len(document.participant_ranks) > 1
            for layout in layouts for document in layout.documents
        ), "full-stack case must exercise a dynamic HDP participant subset"
        loss, should_step = runtime.run_step({BYTESCALE_HDP_WAVES_KEY: _to_device(waves, device)})
        assert should_step and torch.isfinite(loss)
        runtime.step_optimizer()
        runtime.close()
        dist.barrier()
        if rank == 0:
            print(f"PASS: HDP Flash full stack {'TP/SP + ' if tp_sp else ''}ZeRO1/checkpoint/GQA")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        print("SKIP: requires four CUDA devices")
        return
    reason = flash_attn_block_fallback_reason(torch.empty((1, 4, 2, 16), device="cuda", dtype=torch.bfloat16))
    if reason is not None:
        print(f"SKIP: {reason}")
        return
    mp.spawn(_run_case, args=(False,), nprocs=2, join=True)
    mp.spawn(_run_case, args=(True,), nprocs=4, join=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
