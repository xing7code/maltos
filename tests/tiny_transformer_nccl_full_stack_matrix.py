from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import tiny_transformer_ep_full_stack_equivalence as ep_full_eq
import tiny_transformer_ep_full_stack_resume as ep_full_resume
import tiny_transformer_full_stack_equivalence as full_eq
import tiny_transformer_full_stack_resume as full_resume


CaseRunner = Callable[[int, argparse.Namespace, torch.device | None], None]


@dataclass(frozen=True)
class MatrixCase:
    name: str
    runner: CaseRunner
    args: argparse.Namespace
    needs_checkpoint_dir: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29690)
    parser.add_argument("--case-filter", type=str, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def _full_eq_args(**overrides: object) -> argparse.Namespace:
    base = dict(
        world_size=8,
        dp_size=2,
        pp_size=2,
        cp_size=1,
        tp_size=2,
        pp_microbatches=2,
        pp_schedule="afab",
        zero_stage=3,
        cp_attn_core="all_gather_kv",
        master_addr="127.0.0.1",
        master_port=29569,
        backend="nccl",
        batch_size=8,
        seq_len=32,
        seed=42,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _full_resume_args(**overrides: object) -> argparse.Namespace:
    base = dict(
        world_size=8,
        dp_size=2,
        pp_size=2,
        cp_size=1,
        tp_size=2,
        pp_microbatches=2,
        pp_schedule="afab",
        grad_accum_steps=1,
        cp_attn_core="all_gather_kv",
        master_addr="127.0.0.1",
        master_port=29630,
        backend="nccl",
        global_batch_size=8,
        seq_len=32,
        seed=42,
        checkpoint_dir=None,
        zero_stage=3,
        disable_precision=False,
        disable_grad_clip=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _ep_full_eq_args(**overrides: object) -> argparse.Namespace:
    base = dict(
        world_size=8,
        dp_size=2,
        pp_size=2,
        cp_size=2,
        tp_size=1,
        ep_size=4,
        pp_microbatches=2,
        pp_schedule="afab",
        zero_stage=3,
        cp_attn_core="all_gather_kv",
        reuse_tp_for_ep=True,
        reuse_cp_for_ep=True,
        master_addr="127.0.0.1",
        master_port=29644,
        backend="nccl",
        batch_size=8,
        seq_len=32,
        seed=42,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _ep_full_resume_args(**overrides: object) -> argparse.Namespace:
    base = dict(
        world_size=8,
        dp_size=2,
        pp_size=2,
        cp_size=2,
        tp_size=1,
        ep_size=4,
        pp_microbatches=2,
        pp_schedule="afab",
        grad_accum_steps=1,
        cp_attn_core="all_gather_kv",
        reuse_tp_for_ep=True,
        reuse_cp_for_ep=True,
        master_addr="127.0.0.1",
        master_port=29635,
        backend="nccl",
        global_batch_size=8,
        seq_len=32,
        seed=42,
        checkpoint_dir=None,
        zero_stage=3,
        disable_precision=False,
        disable_grad_clip=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _build_cases(args: argparse.Namespace) -> list[MatrixCase]:
    cases: list[MatrixCase] = []

    # A
    for z in (0, 1, 2, 3):
        cases.append(MatrixCase(
            name=f"A/full_eq/z{z}",
            runner=full_eq.run_case,
            args=_full_eq_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=2, tp_size=1,
                zero_stage=z, cp_attn_core="ring",
            ),
        ))
    cases.append(MatrixCase(
        name="A/full_eq/1f1b_z3",
        runner=full_eq.run_case,
        args=_full_eq_args(
            backend=args.backend,
            world_size=args.world_size,
            dp_size=2, pp_size=2, cp_size=2, tp_size=1,
            pp_schedule="1f1b", zero_stage=3, cp_attn_core="ring",
        ),
    ))
    cases.extend([
        MatrixCase(
            name="A/full_resume/acc1",
            runner=full_resume.run_case,
            args=_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=2, tp_size=1,
                grad_accum_steps=1, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
        MatrixCase(
            name="A/full_resume/1f1b_acc1",
            runner=full_resume.run_case,
            args=_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=2, tp_size=1,
                pp_schedule="1f1b", grad_accum_steps=1, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
        MatrixCase(
            name="A/full_resume/acc2",
            runner=full_resume.run_case,
            args=_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=2, tp_size=1,
                grad_accum_steps=2, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
        MatrixCase(
            name="A/full_resume/1f1b_acc2",
            runner=full_resume.run_case,
            args=_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=2, tp_size=1,
                pp_schedule="1f1b", grad_accum_steps=2, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
    ])
    for z in (0, 1, 2, 3):
        cases.append(MatrixCase(
            name=f"A/ep_full_eq/z{z}",
            runner=ep_full_eq.run_case,
            args=_ep_full_eq_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=2, tp_size=1, ep_size=4,
                zero_stage=z, cp_attn_core="ring",
            ),
        ))
    for z in (1, 2, 3):
        cases.append(MatrixCase(
            name=f"A/ep_full_eq/1f1b_z{z}",
            runner=ep_full_eq.run_case,
            args=_ep_full_eq_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=2, tp_size=1, ep_size=4,
                pp_schedule="1f1b", zero_stage=z, cp_attn_core="ring",
            ),
        ))
    cases.extend([
        MatrixCase(
            name="A/ep_full_resume/z1_acc1",
            runner=ep_full_resume.run_case,
            args=_ep_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=2, tp_size=1, ep_size=4,
                zero_stage=1, grad_accum_steps=1, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
        MatrixCase(
            name="A/ep_full_resume/z2_acc1",
            runner=ep_full_resume.run_case,
            args=_ep_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=2, tp_size=1, ep_size=4,
                zero_stage=2, grad_accum_steps=1, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
        MatrixCase(
            name="A/ep_full_resume/z3_acc1",
            runner=ep_full_resume.run_case,
            args=_ep_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=2, tp_size=1, ep_size=4,
                grad_accum_steps=1, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
        MatrixCase(
            name="A/ep_full_resume/z1_1f1b_acc1",
            runner=ep_full_resume.run_case,
            args=_ep_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=2, tp_size=1, ep_size=4,
                zero_stage=1, pp_schedule="1f1b", grad_accum_steps=1, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
        MatrixCase(
            name="A/ep_full_resume/z2_1f1b_acc1",
            runner=ep_full_resume.run_case,
            args=_ep_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=2, tp_size=1, ep_size=4,
                zero_stage=2, pp_schedule="1f1b", grad_accum_steps=1, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
        MatrixCase(
            name="A/ep_full_resume/z3_1f1b_acc1",
            runner=ep_full_resume.run_case,
            args=_ep_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=2, tp_size=1, ep_size=4,
                pp_schedule="1f1b", grad_accum_steps=1, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
        MatrixCase(
            name="A/ep_full_resume/z1_acc2",
            runner=ep_full_resume.run_case,
            args=_ep_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=2, tp_size=1, ep_size=4,
                zero_stage=1, grad_accum_steps=2, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
        MatrixCase(
            name="A/ep_full_resume/z2_acc2",
            runner=ep_full_resume.run_case,
            args=_ep_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=2, tp_size=1, ep_size=4,
                zero_stage=2, grad_accum_steps=2, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
        MatrixCase(
            name="A/ep_full_resume/z3_acc2",
            runner=ep_full_resume.run_case,
            args=_ep_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=2, tp_size=1, ep_size=4,
                grad_accum_steps=2, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
    ])

    # B
    for z in (0, 1, 2, 3):
        cases.append(MatrixCase(
            name=f"B/full_eq/z{z}",
            runner=full_eq.run_case,
            args=_full_eq_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=1, tp_size=2,
                zero_stage=z,
            ),
        ))
    cases.append(MatrixCase(
        name="B/full_eq/1f1b_z3",
        runner=full_eq.run_case,
        args=_full_eq_args(
            backend=args.backend,
            world_size=args.world_size,
            dp_size=2, pp_size=2, cp_size=1, tp_size=2,
            pp_schedule="1f1b", zero_stage=3,
        ),
    ))
    for schedule, accum in (("afab", 1), ("1f1b", 1), ("afab", 2), ("1f1b", 2)):
        cases.append(MatrixCase(
            name=f"B/full_resume/{schedule}_acc{accum}",
            runner=full_resume.run_case,
            args=_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=1, tp_size=2,
                pp_schedule=schedule, grad_accum_steps=accum,
            ),
            needs_checkpoint_dir=True,
        ))
    for z in (0, 1, 2, 3):
        cases.append(MatrixCase(
            name=f"B/ep_full_eq/z{z}",
            runner=ep_full_eq.run_case,
            args=_ep_full_eq_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=1, tp_size=2, ep_size=4,
                zero_stage=z,
            ),
        ))
    cases.append(MatrixCase(
        name="B/ep_full_eq/1f1b_z3",
        runner=ep_full_eq.run_case,
        args=_ep_full_eq_args(
            backend=args.backend,
            world_size=args.world_size,
            dp_size=2, pp_size=2, cp_size=1, tp_size=2, ep_size=4,
            pp_schedule="1f1b", zero_stage=3,
        ),
    ))
    for schedule, accum in (("afab", 1), ("1f1b", 1), ("afab", 2)):
        cases.append(MatrixCase(
            name=f"B/ep_full_resume/{schedule}_acc{accum}",
            runner=ep_full_resume.run_case,
            args=_ep_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=2, cp_size=1, tp_size=2, ep_size=4,
                pp_schedule=schedule, grad_accum_steps=accum,
            ),
            needs_checkpoint_dir=True,
        ))

    # C
    for z in (0, 1, 2, 3):
        cases.append(MatrixCase(
            name=f"C/full_eq/z{z}",
            runner=full_eq.run_case,
            args=_full_eq_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=1, cp_size=2, tp_size=2,
                zero_stage=z, cp_attn_core="ring",
            ),
        ))
    for accum in (1, 2):
        cases.append(MatrixCase(
            name=f"C/full_resume/acc{accum}",
            runner=full_resume.run_case,
            args=_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=1, cp_size=2, tp_size=2,
                grad_accum_steps=accum, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ))
    for z in (0, 1, 2, 3):
        cases.append(MatrixCase(
            name=f"C/ep_full_eq/z{z}",
            runner=ep_full_eq.run_case,
            args=_ep_full_eq_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=1, cp_size=2, tp_size=2, ep_size=8,
                zero_stage=z, cp_attn_core="ring",
            ),
        ))
    for accum in (1, 2):
        cases.append(MatrixCase(
            name=f"C/ep_full_resume/acc{accum}",
            runner=ep_full_resume.run_case,
            args=_ep_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=2, pp_size=1, cp_size=2, tp_size=2, ep_size=8,
                grad_accum_steps=accum, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ))

    # D
    cases.extend([
        MatrixCase(
            name="D/full_eq/z0",
            runner=full_eq.run_case,
            args=_full_eq_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=1, pp_size=2, cp_size=2, tp_size=2,
                zero_stage=0, cp_attn_core="ring",
            ),
        ),
        MatrixCase(
            name="D/full_eq/1f1b_z0",
            runner=full_eq.run_case,
            args=_full_eq_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=1, pp_size=2, cp_size=2, tp_size=2,
                pp_schedule="1f1b", zero_stage=0, cp_attn_core="ring",
            ),
        ),
        MatrixCase(
            name="D/full_resume/acc1",
            runner=full_resume.run_case,
            args=_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=1, pp_size=2, cp_size=2, tp_size=2,
                zero_stage=0, grad_accum_steps=1, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
        MatrixCase(
            name="D/full_resume/1f1b_acc1",
            runner=full_resume.run_case,
            args=_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=1, pp_size=2, cp_size=2, tp_size=2,
                zero_stage=0, pp_schedule="1f1b", grad_accum_steps=1, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
        MatrixCase(
            name="D/full_resume/acc2",
            runner=full_resume.run_case,
            args=_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=1, pp_size=2, cp_size=2, tp_size=2,
                zero_stage=0, grad_accum_steps=2, global_batch_size=4, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
        MatrixCase(
            name="D/ep_full_eq/z0",
            runner=ep_full_eq.run_case,
            args=_ep_full_eq_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=1, pp_size=2, cp_size=2, tp_size=2, ep_size=4,
                zero_stage=0, cp_attn_core="ring",
            ),
        ),
        MatrixCase(
            name="D/ep_full_eq/1f1b_z0",
            runner=ep_full_eq.run_case,
            args=_ep_full_eq_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=1, pp_size=2, cp_size=2, tp_size=2, ep_size=4,
                pp_schedule="1f1b", zero_stage=0, cp_attn_core="ring",
            ),
        ),
        MatrixCase(
            name="D/ep_full_resume/acc1",
            runner=ep_full_resume.run_case,
            args=_ep_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=1, pp_size=2, cp_size=2, tp_size=2, ep_size=4,
                zero_stage=0, grad_accum_steps=1, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
        MatrixCase(
            name="D/ep_full_resume/1f1b_acc1",
            runner=ep_full_resume.run_case,
            args=_ep_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=1, pp_size=2, cp_size=2, tp_size=2, ep_size=4,
                zero_stage=0, pp_schedule="1f1b", grad_accum_steps=1, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
        MatrixCase(
            name="D/ep_full_resume/acc2",
            runner=ep_full_resume.run_case,
            args=_ep_full_resume_args(
                backend=args.backend,
                world_size=args.world_size,
                dp_size=1, pp_size=2, cp_size=2, tp_size=2, ep_size=4,
                zero_stage=0, grad_accum_steps=2, global_batch_size=4, cp_attn_core="ring",
            ),
            needs_checkpoint_dir=True,
        ),
    ])

    if args.case_filter:
        cases = [case for case in cases if args.case_filter in case.name]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise ValueError("no matrix cases selected")
    return cases


def _make_checkpoint_dir(case_name: str) -> str:
    obj = [None]
    if dist.get_rank() == 0:
        safe_name = case_name.replace("/", "_")
        obj[0] = tempfile.mkdtemp(prefix=f"{safe_name}_")
    dist.broadcast_object_list(obj)
    assert isinstance(obj[0], str)
    return obj[0]


def _cleanup_checkpoint_dir(path: str | None) -> None:
    if path is None:
        return
    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(path, ignore_errors=True)
    dist.barrier()


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    if args.world_size != 8:
        raise ValueError(f"this runner currently expects world_size=8, got {args.world_size}")
    device: torch.device | None = None
    if args.backend == "nccl":
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
        device_id=device,
    )
    try:
        cases = _build_cases(args)
        if rank == 0:
            print(f"Selected {len(cases)} NCCL full-stack cases")
        for idx, case in enumerate(cases, start=1):
            case_args = argparse.Namespace(**vars(case.args))
            checkpoint_dir: str | None = None
            if case.needs_checkpoint_dir:
                checkpoint_dir = _make_checkpoint_dir(case.name)
                case_args.checkpoint_dir = checkpoint_dir
            if rank == 0:
                print(f"=== [{idx}/{len(cases)}] {case.name} ===")
            dist.barrier()
            try:
                case.runner(rank, case_args, device)
            finally:
                _cleanup_checkpoint_dir(checkpoint_dir)
                if device is not None:
                    torch.cuda.empty_cache()
            dist.barrier()
        if rank == 0:
            print("NCCL full-stack matrix PASS")
    finally:
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
