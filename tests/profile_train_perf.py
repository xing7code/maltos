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
from dataclasses import replace
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from helpers import causal_lm_batch
from models import LlamaConfig, LlamaForCausalLMTpSp, TinyMoETransformerTpSp
from models.activation_checkpointing import ActivationCheckpointConfig
from models.llama import LlamaRMSNorm
from models.tiny_transformer import RmsNorm
from parallel import ContextParallelAttentionCoreType, ParallelPlan
from parallel.schedule import PipelineScheduleConfig
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.plugins.cp import ContextParallelPlugin
from runtime.plugins.ep import ExpertParallelPlugin
from runtime.plugins.pp import PipelineParallelPlugin
from runtime.plugins.precision import PrecisionPlugin
from runtime.plugins.sp import SequenceParallelPlugin
from runtime.plugins.tp import TensorParallelPlugin
from runtime.plugins.zero1 import Zero1Plugin
from runtime.plugins.zero3 import Zero3Plugin


# ── model presets ────────────────────────────────────────────────────────────

# ~1B class profile target using SDPA flash + activation checkpointing.
_LLAMA_1B = LlamaConfig(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=8192,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=4,
    max_position_embeddings=8192,
    attention_backend="sdpa_flash",
    activation_checkpointing=ActivationCheckpointConfig(enabled=True, every_n_layers=1),
)

# Lighter TP target for PCIe tensor-parallel profile.
_LLAMA_350M = LlamaConfig(
    vocab_size=32000,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=4,
    max_position_embeddings=8192,
    attention_backend="sdpa_flash",
    activation_checkpointing=ActivationCheckpointConfig(enabled=True, every_n_layers=1),
)

# Moderate MoE profile target. Top-1 routing keeps per-token compute near dense,
# while expert params are sharded by EP to surface expert-parallel communication.
_MOE_MED = dict(dim=768, n_heads=12, n_kv_heads=4, hidden_size=3072,
                eps=1e-5, n_layers=16, vocab_size=32000, max_seq_len=4096, num_experts=8)

_ZERO3_WRAP = {torch.nn.Linear, torch.nn.Embedding, RmsNorm, LlamaRMSNorm}


@dataclass(frozen=True)
class PerfCase:
    desc: str
    dp: int; pp: int; tp: int; cp: int
    ep: int
    zero_stage: int
    seq_len: int; batch_per_dp: int
    pp_microbatches: int; pp_schedule: str
    use_sp: bool; cp_attn_core: str
    model: object
    use_ep: bool = False
    grad_clip: float | None = 1.0

    @property
    def total_toks_per_step(self) -> int:
        return self.dp * self.batch_per_dp * self.seq_len


def _compute_dtype_for_name(name: str) -> torch.dtype | None:
    if name == "fp32":
        return None
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported precision={name}")


def _model_vocab_size(model_cfg: object) -> int:
    if isinstance(model_cfg, LlamaConfig):
        return int(model_cfg.vocab_size)
    return int(model_cfg["vocab_size"])


def _model_summary(model_cfg: object) -> str:
    if isinstance(model_cfg, LlamaConfig):
        return (
            f"{model_cfg.num_hidden_layers}L × hidden={model_cfg.hidden_size}, "
            f"intermediate={model_cfg.intermediate_size}, heads={model_cfg.num_attention_heads}"
        )
    return (
        f"{model_cfg['n_layers']}L × dim={model_cfg['dim']}, "
        f"hidden={model_cfg['hidden_size']}"
    )


_CASES: dict[str, PerfCase] = {
    "A": PerfCase(
        desc="dp=8  pp=1  tp=1  cp=1  zero=3  (pure ZeRO-3 baseline)",
        dp=8, pp=1, tp=1, cp=1, ep=1, zero_stage=3,
        seq_len=2048, batch_per_dp=4,
        pp_microbatches=1, pp_schedule="afab",
        use_sp=False, cp_attn_core="all_gather_kv",
        model=_LLAMA_1B,
    ),
    "B": PerfCase(
        desc="dp=4  pp=2  tp=1  cp=1  zero=3",
        dp=4, pp=2, tp=1, cp=1, ep=1, zero_stage=3,
        seq_len=2048, batch_per_dp=6,
        pp_microbatches=4, pp_schedule="afab",
        use_sp=False, cp_attn_core="all_gather_kv",
        model=_LLAMA_1B,
    ),
    "C": PerfCase(
        desc="dp=2  pp=4  tp=1  cp=1  zero=3  (1f1b)",
        dp=2, pp=4, tp=1, cp=1, ep=1, zero_stage=3,
        seq_len=2048, batch_per_dp=6,
        pp_microbatches=8, pp_schedule="1f1b",
        use_sp=False, cp_attn_core="all_gather_kv",
        model=_LLAMA_1B,
    ),
    "D": PerfCase(
        desc="dp=4  pp=1  tp=2  cp=1  zero=1  (TP cost on PCIe)",
        dp=4, pp=1, tp=2, cp=1, ep=1, zero_stage=1,
        seq_len=2048, batch_per_dp=4,
        pp_microbatches=1, pp_schedule="afab",
        use_sp=True, cp_attn_core="all_gather_kv",
        model=_LLAMA_350M,
    ),
    "E": PerfCase(
        desc="dp=2  pp=2  tp=2  cp=1  zero=3  (3D: PP+TP+ZeRO-3)",
        dp=2, pp=2, tp=2, cp=1, ep=1, zero_stage=3,
        seq_len=2048, batch_per_dp=4,
        pp_microbatches=4, pp_schedule="afab",
        use_sp=True, cp_attn_core="all_gather_kv",
        model=_LLAMA_1B,
    ),
    "F": PerfCase(
        desc="dp=4  pp=1  tp=1  cp=2  zero=3  (CP2 ring-attn  seq=4096)",
        dp=4, pp=1, tp=1, cp=2, ep=1, zero_stage=3,
        seq_len=4096, batch_per_dp=1,
        pp_microbatches=1, pp_schedule="afab",
        use_sp=False, cp_attn_core="ring",
        model=_LLAMA_1B,
    ),
    "G": PerfCase(
        desc="dp=8  pp=1  tp=1  cp=1  ep=4  zero=3  (MoE EP baseline)",
        dp=8, pp=1, tp=1, cp=1, ep=4, zero_stage=3,
        seq_len=1024, batch_per_dp=2,
        pp_microbatches=1, pp_schedule="afab",
        use_sp=False, cp_attn_core="all_gather_kv",
        model=_MOE_MED,
        use_ep=True,
    ),
}


# ── build runtime ─────────────────────────────────────────────────────────────

def _build_core(cfg: PerfCase, device: torch.device) -> RuntimeCore:
    model = TinyMoETransformerTpSp(**cfg.model) if cfg.use_ep else LlamaForCausalLMTpSp(cfg.model)

    plugins = []
    if cfg.tp > 1:
        plugins += [TensorParallelPlugin()]
        if cfg.use_sp:
            plugins += [SequenceParallelPlugin()]
    if cfg.cp > 1:
        plugins += [ContextParallelPlugin()]
    if cfg.pp > 1:
        plugins += [PipelineParallelPlugin(schedule=cfg.pp_schedule)]
    if cfg.use_ep:
        plugins += [ExpertParallelPlugin()]
    if cfg.zero_stage == 1:
        plugins += [Zero1Plugin(bucket_mb_size=32)]
    elif cfg.zero_stage == 3:
        plugins += [Zero3Plugin(wrap_cls=_ZERO3_WRAP)]
    plugins += [PrecisionPlugin(compute_dtype=_compute_dtype_for_name(_ACTIVE_PRECISION))]

    return RuntimeCore(
        mesh=MeshConfig(dp=cfg.dp, tp=cfg.tp, pp=cfg.pp, cp=cfg.cp, ep=cfg.ep),
        plan=ParallelPlan(
            pp_schedule=PipelineScheduleConfig(microbatches=cfg.pp_microbatches),
            cp_attn_core=ContextParallelAttentionCoreType(cfg.cp_attn_core),
        ),
        model=model,
        optimizer_factory=lambda params: torch.optim.AdamW(params, lr=1e-4),
        plugins=plugins,
        grad_clip_max_norm=cfg.grad_clip,
        device=device,
    )


_ACTIVE_PRECISION = "bf16"


def _resolve_case(base: PerfCase, args: argparse.Namespace) -> PerfCase:
    updates: dict[str, object] = {}
    if args.batch_per_dp_override is not None:
        updates["batch_per_dp"] = args.batch_per_dp_override
    if args.seq_len_override is not None:
        updates["seq_len"] = args.seq_len_override
    if args.pp_microbatches_override is not None:
        updates["pp_microbatches"] = args.pp_microbatches_override
    if args.disable_activation_checkpointing and isinstance(base.model, LlamaConfig):
        updates["model"] = LlamaConfig(
            vocab_size=base.model.vocab_size,
            hidden_size=base.model.hidden_size,
            intermediate_size=base.model.intermediate_size,
            num_hidden_layers=base.model.num_hidden_layers,
            num_attention_heads=base.model.num_attention_heads,
            num_key_value_heads=base.model.num_key_value_heads,
            max_position_embeddings=base.model.max_position_embeddings,
            rms_norm_eps=base.model.rms_norm_eps,
            rope_theta=base.model.rope_theta,
            tie_word_embeddings=base.model.tie_word_embeddings,
            attention_backend=base.model.attention_backend,
            activation_checkpointing=ActivationCheckpointConfig(enabled=False),
        )
    if not updates:
        return base
    return replace(base, **updates)


# ── worker ────────────────────────────────────────────────────────────────────

def _run_worker(rank: int, args: argparse.Namespace) -> None:
    global _ACTIVE_PRECISION
    _ACTIVE_PRECISION = args.precision
    cfg = _resolve_case(_CASES[args.case], args)

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
    core: RuntimeCore | None = None
    try:
        if rank == 0:
            print(
                f"[case {args.case}] starting: {cfg.desc} | precision={args.precision} | "
                f"seq={cfg.seq_len} batch_per_dp={cfg.batch_per_dp}",
                flush=True,
            )
        core = _build_core(cfg, device)
        core.setup()
        core.model.train()

        dp_rank = rank // (cfg.tp * cfg.pp * cfg.cp)
        local_bs = cfg.batch_per_dp
        torch.manual_seed(42 + dp_rank)
        fake_tokens = torch.randint(
            0, _model_vocab_size(cfg.model), (local_bs, cfg.seq_len), device=device
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

        local_peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
        local_avg_ms = sum(step_times) / len(step_times) * 1e3
        local_min_ms = min(step_times) * 1e3
        local_max_ms = max(step_times) * 1e3

        metrics = torch.tensor(
            [local_avg_ms, local_min_ms, local_max_ms, local_peak_mem_gb],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(metrics[0:1], op=dist.ReduceOp.MAX)
        dist.all_reduce(metrics[1:2], op=dist.ReduceOp.MIN)
        dist.all_reduce(metrics[2:3], op=dist.ReduceOp.MAX)
        dist.all_reduce(metrics[3:4], op=dist.ReduceOp.MAX)
        avg_ms = float(metrics[0].item())
        min_ms = float(metrics[1].item())
        max_ms = float(metrics[2].item())
        peak_mem_gb = float(metrics[3].item())
        toks_per_sec = cfg.total_toks_per_step / (avg_ms / 1e3)

        if rank == 0:
            lines = [
                f"{'─'*60}",
                f"Case {args.case}: {cfg.desc}",
                f"Precision   : {args.precision}",
                f"Model       : {_model_summary(cfg.model)}",
                f"Tokens/step : {cfg.total_toks_per_step:,}  (dp={cfg.dp} × batch={cfg.batch_per_dp} × seq={cfg.seq_len})",
                f"Step time   : {avg_ms:.1f} ms  (global max-rank avg; min={min_ms:.1f}, max={max_ms:.1f})",
                f"Throughput  : {toks_per_sec:,.0f} tok/s",
                f"Peak VRAM   : {peak_mem_gb:.2f} GB  (global max rank)",
            ]
            if args.profile:
                trace_root = Path(args.output_dir)
                lines.append(f"Trace       : {trace_root}/trace_case_{args.case}_rank*/")
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
                f"  precision={args.precision}"
                f"  [{cfg.desc}]\n"
            )
            with (out_dir / "summary.txt").open("a") as fh:
                fh.write(summary_line)

    except BaseException as exc:
        print(f"[rank {rank}] case {args.case} failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        if core is not None:
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
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-per-dp-override", type=int, default=None)
    parser.add_argument("--seq-len-override", type=int, default=None)
    parser.add_argument("--pp-microbatches-override", type=int, default=None)
    parser.add_argument("--disable-activation-checkpointing", action="store_true")
    parser.add_argument("--output-dir", type=str, default="profiles")
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_cases:
        for k, v in _CASES.items():
            print(f"  {k}  {v.desc}")
            print(f"     tokens/step={v.total_toks_per_step:,}  model={_model_summary(v.model)}")
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
