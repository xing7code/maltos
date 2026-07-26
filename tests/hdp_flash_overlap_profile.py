"""CUDA profiler validation for HDP async Flash Ring.

Run on a three-GPU NCCL host:
  BYTESCALE_HDP_PROFILE=1 PYTHONPATH=. .venv/bin/python tests/hdp_flash_overlap_profile.py

The trace is written to /tmp and must contain both FlashAttention kernels and
NCCL send/recv operations.  The script reports, but does not fabricate, kernel
interval overlap when the installed PyTorch profiler exposes those intervals.
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import ProfilerActivity, profile

from runtime.layers.cp_functional import dynamic_flash_ring_attention
from runtime.layers.flash_utils import flash_attn_block_fallback_reason


_PORT = 29684


def _worker(rank: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{_PORT}", rank=rank, world_size=3)
    try:
        device = torch.device("cuda", rank)
        q = torch.randn(1, 4, 256, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, 2, 256, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, 2, 256, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        positions = (torch.arange(256, device=device) + rank * 256).unsqueeze(0)
        participants = (0, 2) if rank != 1 else (1,)
        # Warmup initializes FlashAttention and pinned double buffers outside the trace.
        out = dynamic_flash_ring_attention(
            q, k, v, positions=positions, sequence_ids=None,
            group=dist.group.WORLD, participant_ranks=participants, module_id=917,
        )
        out.float().sum().backward()
        q.grad = k.grad = v.grad = None
        dist.barrier()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            out = dynamic_flash_ring_attention(
                q, k, v, positions=positions, sequence_ids=None,
                group=dist.group.WORLD, participant_ranks=participants, module_id=917,
            )
            out.float().sum().backward()
            torch.cuda.synchronize(device)
        trace = Path(f"/tmp/hdp_flash_overlap_rank{rank}.json")
        prof.export_chrome_trace(str(trace))
        names = [event.key.lower() for event in prof.key_averages()]
        has_flash = any("flash" in name for name in names)
        has_p2p = any("nccl" in name or "send" in name or "recv" in name for name in names)
        needs_p2p = len(participants) > 1
        if not has_flash or (needs_p2p and not has_p2p):
            raise AssertionError(f"trace misses required events: flash={has_flash} p2p={has_p2p}; {names}")
        semantic = {
            event.key: event.self_cpu_time_total
            for event in prof.key_averages()
            if event.key.startswith("maltos::hdp.flash")
        }
        required_spans = {"maltos::hdp.flash.forward.block"}
        if needs_p2p:
            required_spans.add("maltos::hdp.flash.p2p.launch")
        if not required_spans <= semantic.keys():
            raise AssertionError(f"missing HDP profiler spans: {semantic}")
        events = json.loads(trace.read_text()).get("traceEvents", [])
        kernels = [
            event for event in events
            if event.get("ph") == "X" and event.get("dur", 0) > 0
        ]
        flash_kernels = [event for event in kernels if "flash" in event.get("name", "").lower()]
        p2p_kernels = [
            event for event in kernels
            if "nccl" in event.get("name", "").lower()
            or "sendrecv" in event.get("name", "").lower()
        ]
        overlap = any(
            max(flash["ts"], p2p["ts"]) < min(flash["ts"] + flash["dur"], p2p["ts"] + p2p["dur"])
            for flash in flash_kernels for p2p in p2p_kernels
        )
        if not overlap and needs_p2p:
            raise AssertionError("trace has Flash/NCCL kernels but no overlapping device intervals")
        if not needs_p2p and has_p2p:
            raise AssertionError("single-participant Flash fast path unexpectedly emitted P2P activity")
        if rank == 0:
            print(f"PASS: Flash/NCCL P2P overlap captured; traces: /tmp/hdp_flash_overlap_rank*.json")
            print(f"HDP CPU span self times (us): {semantic}")
    finally:
        dist.destroy_process_group()


def main() -> None:
    if os.environ.get("BYTESCALE_HDP_PROFILE") != "1":
        print("SKIP: set BYTESCALE_HDP_PROFILE=1 to run the expensive NCCL profiler validation")
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        print("SKIP: requires three CUDA devices")
        return
    probe = torch.empty((1, 4, 2, 64), device="cuda", dtype=torch.bfloat16)
    reason = flash_attn_block_fallback_reason(probe)
    if reason is not None:
        print(f"SKIP: {reason}")
        return
    mp.spawn(_worker, nprocs=3, join=True)


if __name__ == "__main__":
    main()
