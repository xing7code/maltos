"""Parallelism performance profiler for 8x GPU.

Runs a few timed steps with a realistic ~1.3B (or 300M for TP) model on
synthetic data, then reports step_time, tokens/sec, and peak GPU memory.
Optionally saves a torch.profiler Chrome trace per rank.

Available cases (world=8, see _CASES dict below):
  A  dp=8  pp=1  tp=1  cp=1  z=3  seq=2048   — pure ZeRO-3 baseline
  B  dp=4  pp=2  tp=1  cp=1  z=3  seq=2048   — PP2 + ZeRO-3
  C  dp=2  pp=4  tp=1  cp=1  z=3  seq=2048   — PP4 + ZeRO-3 (1f1b)
  D  dp=4  pp=1  tp=2  cp=1  z=1  seq=2048   — TP2 + ZeRO-1 (PCIe TP cost)
  E  dp=2  pp=2  tp=2  cp=1  z=3  seq=2048   — 3D: PP2+TP2+ZeRO-3
  F  dp=4  pp=1  tp=1  cp=2  z=3  seq=8192   — CP2 ring-attn long-seq

Usage:
  PYTHONPATH=. BACKEND=nccl .venv/bin/python tests/profile_train_perf.py --case A
  PYTHONPATH=. BACKEND=nccl .venv/bin/python tests/profile_train_perf.py --case A --profile
  PYTHONPATH=. BACKEND=nccl .venv/bin/python tests/profile_train_perf.py --list-cases
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from helpers import causal_lm_batch
from models import TinyTransformerTpSp
from models.tiny_transformer import RmsNorm
from parallel import ContextParallelAttentionCoreType, ParallelPlan
from parallel.schedule import PipelineScheduleConfig
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.plugins.cp import ContextParallelPlugin
from runtime.plugins.pp import PipelineParallelPlugin
from runtime.plugins.sp import SequenceParallelPlugin
from runtime.plugins.tp import TensorParallelPlugin
from runtime.plugins.zero1 import Zero1Plugin
from runtime.plugins.zero3 import Zero3Plugin


# ── model presets ────────────────────────────────────────────────────────────

# ~1.3B params (LLaMA-1.3B-ish)
_1B = dict(dim=2048, n_heads=16, n_kv_heads=4, hidden_size=8192,
           eps=1e-5, n_layers=24, vocab_size=32000, max_seq_len=8192)

# ~350M params — lighter for TP cases where param memory is 2× replicated
_350M = dict(dim=1024, n_heads=16, n_kv_heads=4, hidden_size=4096,
             eps=1e-5, n_layers=24, vocab_size=32000, max_seq_len=8192)

_ZERO3_WRAP = {torch.nn.Linear, torch.nn.Embedding, RmsNorm}


@dataclass(frozen=True)
class PerfCase:
    desc: str
    dp: int; pp: int; tp: int; cp: int
    zero_stage: int
    seq_len: int; batch_per_dp: int
    pp_microbatches: int; pp_schedule: str
    use_sp: bool; cp_attn_core: str
    model: dict
    grad_clip: float | None = 1.0

    @property
    def total_toks_per_step(self) -> int:
        return self.dp * self.batch_per_dp * self.seq_len


_CASES: dict[str, PerfCase] = {
    "A": PerfCase(
        desc="dp=8  pp=1  tp=1  cp=1  zero=3  (pure ZeRO-3 baseline)",
        dp=8, pp=1, tp=1, cp=1, zero_stage=3,
        seq_len=2048, batch_per_dp=4,
        pp_microbatches=1, pp_schedule="afab",
        use_sp=False, cp_attn_core="all_gather_kv",
        model=_1B,
    ),
    "B": PerfCase(
        desc="dp=4  pp=2  tp=1  cp=1  zero=3",
        dp=4, pp=2, tp=1, cp=1, zero_stage=3,
        seq_len=2048, batch_per_dp=8,
        pp_microbatches=4, pp_schedule="afab",
        use_sp=False, cp_attn_core="all_gather_kv",
        model=_1B,
    ),
    "C": PerfCase(
        desc="dp=2  pp=4  tp=1  cp=1  zero=3  (1f1b)",
        dp=2, pp=4, tp=1, cp=1, zero_stage=3,
        seq_len=2048, batch_per_dp=8,
        pp_microbatches=8, pp_schedule="1f1b",
        use_sp=False, cp_attn_core="all_gather_kv",
        model=_1B,
    ),
    "D": PerfCase(
        desc="dp=4  pp=1  tp=2  cp=1  zero=1  (TP cost on PCIe)",
        dp=4, pp=1, tp=2, cp=1, zero_stage=1,
        seq_len=2048, batch_per_dp=4,
        pp_microbatches=1, pp_schedule="afab",
        use_sp=True, cp_attn_core="all_gather_kv",
        model=_350M,
    ),
    "E": PerfCase(
        desc="dp=2  pp=2  tp=2  cp=1  zero=3  (3D: PP+TP+ZeRO-3)",
        dp=2, pp=2, tp=2, cp=1, zero_stage=3,
        seq_len=2048, batch_per_dp=8,
        pp_microbatches=4, pp_schedule="afab",
        use_sp=True, cp_attn_core="all_gather_kv",
        model=_1B,
    ),
    "F": PerfCase(
        desc="dp=4  pp=1  tp=1  cp=2  zero=3  (CP2 ring-attn  seq=8192)",
        dp=4, pp=1, tp=1, cp=2, zero_stage=3,
        seq_len=8192, batch_per_dp=2,
        pp_microbatches=1, pp_schedule="afab",
        use_sp=False, cp_attn_core="ring",
        model=_1B,
    ),
}


# ── build runtime ─────────────────────────────────────────────────────────────

def _build_core(cfg: PerfCase, device: torch.device) -> RuntimeCore:
    model = TinyTransformerTpSp(**cfg.model)

    plugins = []
    if cfg.tp > 1:
        plugins += [TensorParallelPlugin()]
        if cfg.use_sp:
            plugins += [SequenceParallelPlugin()]
    if cfg.cp > 1:
        plugins += [ContextParallelPlugin()]
    if cfg.pp > 1:
        plugins += [PipelineParallelPlugin(schedule=cfg.pp_schedule)]
    if cfg.zero_stage == 1:
        plugins += [Zero1Plugin(bucket_mb_size=32)]
    elif cfg.zero_stage == 3:
        plugins += [Zero3Plugin(wrap_cls=_ZERO3_WRAP)]

    return RuntimeCore(
        mesh=MeshConfig(dp=cfg.dp, tp=cfg.tp, pp=cfg.pp, cp=cfg.cp, ep=1),
        plan=ParallelPlan(
            zero_stage=cfg.zero_stage,
            pp_schedule=PipelineScheduleConfig(microbatches=cfg.pp_microbatches),
            cp_attn_core=ContextParallelAttentionCoreType(cfg.cp_attn_core),
        ),
        model=model,
        optimizer_factory=lambda params: torch.optim.AdamW(params, lr=1e-4),
        plugins=plugins,
        grad_clip_max_norm=cfg.grad_clip,
        device=device,
    )


# ── worker ────────────────────────────────────────────────────────────────────

def _run_worker(rank: int, args: argparse.Namespace) -> None:
    cfg = _CASES[args.case]

    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
        device_id=device,
    )
    try:
        core = _build_core(cfg, device)
        core.setup()
        core.model.train()

        dp_rank = rank // (cfg.tp * cfg.pp * cfg.cp)
        local_bs = cfg.batch_per_dp
        torch.manual_seed(42 + dp_rank)
        fake_tokens = torch.randint(
            0, cfg.model["vocab_size"], (local_bs, cfg.seq_len), device=device
        )
        batch = causal_lm_batch(fake_tokens)

        def _step() -> float:
            loss, _ = core.run_step(batch)
            core.step_optimizer()
            torch.cuda.synchronize()
            return float(loss.detach())

        # warmup
        for _ in range(args.warmup):
            _step()
        torch.cuda.reset_peak_memory_stats(device)

        trace_dir = str(Path(args.output_dir) / f"trace_case_{args.case}_rank{rank}")
        if args.profile:
            os.makedirs(trace_dir, exist_ok=True)

        # optional profiler
        prof_ctx = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=args.steps - 1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
                record_shapes=False,
                with_stack=False,
            )
            if args.profile
            else None
        )

        step_times: list[float] = []
        with (prof_ctx if prof_ctx is not None else _null_ctx()):
            for _ in range(args.steps):
                t0 = time.perf_counter()
                _step()
                step_times.append(time.perf_counter() - t0)
                if prof_ctx is not None:
                    prof_ctx.step()

        peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
        avg_ms = sum(step_times) / len(step_times) * 1e3
        toks_per_sec = cfg.total_toks_per_step / (avg_ms / 1e3)

        if rank == 0:
            lines = [
                f"{'─'*60}",
                f"Case {args.case}: {cfg.desc}",
                f"Model       : {cfg.model['n_layers']}L × dim={cfg.model['dim']}, hidden={cfg.model['hidden_size']}",
                f"Tokens/step : {cfg.total_toks_per_step:,}  (dp={cfg.dp} × batch={cfg.batch_per_dp} × seq={cfg.seq_len})",
                f"Step time   : {avg_ms:.1f} ms  (min={min(step_times)*1e3:.1f}, max={max(step_times)*1e3:.1f})",
                f"Throughput  : {toks_per_sec:,.0f} tok/s",
                f"Peak VRAM   : {peak_mem_gb:.2f} GB  (rank 0)",
            ]
            if args.profile:
                trace_dir = Path(args.output_dir) / f"trace_case_{args.case}"
                lines.append(f"Trace       : {trace_dir}/")
            lines.append(f"{'─'*60}")

            output = "\n".join(lines) + "\n"
            print(output)

            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"case_{args.case}.txt").write_text(output)

            summary_line = (
                f"case_{args.case}  step={avg_ms:.1f}ms"
                f"  toks/s={toks_per_sec:,.0f}"
                f"  vram={peak_mem_gb:.2f}GB"
                f"  [{cfg.desc}]\n"
            )
            with (out_dir / "summary.txt").open("a") as fh:
                fh.write(summary_line)

    finally:
        core.close()
        dist.destroy_process_group()


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=list(_CASES), default="A")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29800)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output-dir", type=str, default="profiles")
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_cases:
        for k, v in _CASES.items():
            print(f"  {k}  {v.desc}")
            print(f"     tokens/step={v.total_toks_per_step:,}  model={v.model['n_layers']}L×{v.model['dim']}")
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
