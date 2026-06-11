from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import tiny_transformer_ep_full_stack_equivalence as ep_full_eq
import tiny_transformer_ep_full_stack_resume as ep_full_resume
import tiny_transformer_full_stack_equivalence as full_eq
import tiny_transformer_full_stack_resume as full_resume
from full_stack_matrix_cases import MODULE_SCRIPTS, MatrixCase, build_full_stack_matrix_cases


RUN_CASE_BY_MODULE = {
    "full_eq": full_eq.run_case,
    "full_resume": full_resume.run_case,
    "ep_full_eq": ep_full_eq.run_case,
    "ep_full_resume": ep_full_resume.run_case,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29690)
    parser.add_argument("--case-names", nargs="+", default=None)
    parser.add_argument("--case-file", type=str, default=None)
    parser.add_argument("--case-filter", type=str, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def _resolve_case_names(args: argparse.Namespace) -> list[str] | None:
    case_names: list[str] = []
    if args.case_names:
        case_names.extend(args.case_names)
    if args.case_file is not None:
        path = Path(args.case_file)
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            case_names.append(stripped)
    return case_names or None


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


def _run_merged_worker(rank: int, args: argparse.Namespace) -> None:
    case_names = _resolve_case_names(args)
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
        cases = build_full_stack_matrix_cases(
            backend=args.backend,
            world_size=args.world_size,
            case_names=case_names,
            case_filter=args.case_filter,
            max_cases=args.max_cases,
        )
        if rank == 0:
            print(f"Selected {len(cases)} merged full-stack cases")
        for idx, case in enumerate(cases, start=1):
            case_args = case.to_namespace()
            checkpoint_dir: str | None = None
            if case.needs_checkpoint_dir:
                checkpoint_dir = _make_checkpoint_dir(case.name)
                case_args.checkpoint_dir = checkpoint_dir
            if rank == 0:
                print(f"=== [{idx}/{len(cases)}] {case.name} ===")
            dist.barrier()
            try:
                RUN_CASE_BY_MODULE[case.module_key](rank, case_args, device)
            finally:
                _cleanup_checkpoint_dir(checkpoint_dir)
                if device is not None:
                    torch.cuda.empty_cache()
            dist.barrier()
        if rank == 0:
            print("Merged full-stack matrix PASS")
    finally:
        dist.destroy_process_group()


def _run_subprocess_case(case: MatrixCase) -> None:
    checkpoint_dir: str | None = None
    try:
        cli_args = case.to_cli_args()
        if case.needs_checkpoint_dir:
            safe_name = case.name.replace("/", "_")
            checkpoint_dir = tempfile.mkdtemp(prefix=f"{safe_name}_")
            cli_args.extend(["--checkpoint-dir", checkpoint_dir])
        cmd = [sys.executable, MODULE_SCRIPTS[case.module_key], *cli_args]
        print(f"=== {case.name} ===")
        subprocess.run(cmd, check=True)
    finally:
        if checkpoint_dir is not None:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


def run_subprocess_matrix(args: argparse.Namespace) -> None:
    case_names = _resolve_case_names(args)
    cases = build_full_stack_matrix_cases(
        backend=args.backend,
        world_size=args.world_size,
        case_names=case_names,
        case_filter=args.case_filter,
        max_cases=args.max_cases,
    )
    print(f"Selected {len(cases)} subprocess full-stack cases")
    for case in cases:
        _run_subprocess_case(case)
    print("Subprocess full-stack matrix PASS")


def _print_case_names(args: argparse.Namespace) -> None:
    case_names = _resolve_case_names(args)
    cases = build_full_stack_matrix_cases(
        backend=args.backend,
        world_size=args.world_size,
        case_names=case_names,
        case_filter=args.case_filter,
        max_cases=args.max_cases,
    )
    for case in cases:
        print(case.name)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.list_cases:
        _print_case_names(args)
        return
    if args.backend == "nccl":
        mp.spawn(_run_merged_worker, args=(args,), nprocs=args.world_size, join=True)
    else:
        run_subprocess_matrix(args)


if __name__ == "__main__":
    main()
