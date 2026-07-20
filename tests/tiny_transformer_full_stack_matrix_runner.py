from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import tiny_transformer_ep_full_stack_equivalence as ep_full_eq
import tiny_transformer_ep_full_stack_resume as ep_full_resume
import tiny_transformer_full_stack_equivalence as full_eq
import tiny_transformer_full_stack_resume as full_resume
from full_stack_matrix_cases import MODULE_SCRIPTS, MatrixCase, build_full_stack_matrix_cases
from runtime.buffer_allocator import clear_buffer_pool

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_NOTES_DIR = _REPO_ROOT / "local_notes"
_DEFAULT_WHITELIST = str(_LOCAL_NOTES_DIR / "matrix_whitelist.txt")
_DEFAULT_BLACKLIST = str(_LOCAL_NOTES_DIR / "matrix_blacklist.txt")
_DEFAULT_REPORT_FILE = str(_LOCAL_NOTES_DIR / "matrix_report.log")
_DEFAULT_FAILURES_FILE = str(_LOCAL_NOTES_DIR / "matrix_failures.txt")
_DEFAULT_PASSES_FILE = str(_LOCAL_NOTES_DIR / "matrix_passes.txt")
_PORT_RETRY_LIMIT = 5
_MERGED_CASE_TIMEOUT_SEC = 40.0


RUN_CASE_BY_MODULE = {
    "full_eq": full_eq.run_case,
    "full_resume": full_resume.run_case,
    "ep_full_eq": ep_full_eq.run_case,
    "ep_full_resume": ep_full_resume.run_case,
}


@dataclass(frozen=True)
class CaseResult:
    case: MatrixCase
    ok: bool
    duration_sec: float | None
    returncode: int


class MergedCaseTimeoutError(RuntimeError):
    def __init__(self, case_name: str, timeout_sec: float) -> None:
        super().__init__(f"merged case timed out after {timeout_sec:.1f}s: {case_name}")
        self.case_name = case_name
        self.timeout_sec = timeout_sec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29690)
    parser.add_argument("--case-names", "--whitelist-cases", dest="case_names", nargs="+", default=None)
    parser.add_argument("--blacklist-names", nargs="+", default=None)
    parser.add_argument("--whitelist", "--case-file", dest="whitelist", type=str, default=_DEFAULT_WHITELIST)
    parser.add_argument("--blacklist", type=str, default=_DEFAULT_BLACKLIST)
    parser.add_argument("--case-filter", type=str, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--merge", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--case-timeout-sec", type=float, default=_MERGED_CASE_TIMEOUT_SEC)
    parser.add_argument("--report-file", type=str, default=_DEFAULT_REPORT_FILE)
    parser.add_argument("--failures-file", type=str, default=_DEFAULT_FAILURES_FILE)
    parser.add_argument("--passes-file", type=str, default=_DEFAULT_PASSES_FILE)
    parser.add_argument("--completed-cases-file", type=str, default=None)
    return parser.parse_args()


def _read_case_list(path_str: str | None) -> list[str]:
    if path_str is None:
        return []
    path = Path(path_str)
    if not path.exists():
        return []
    case_names: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        case_names.append(stripped)
    return case_names


def _resolve_case_names(args: argparse.Namespace) -> tuple[list[str] | None, list[str] | None]:
    whitelist: list[str] = []
    blacklist: list[str] = []
    if args.case_names:
        whitelist.extend(args.case_names)
    if args.blacklist_names:
        blacklist.extend(args.blacklist_names)
    whitelist.extend(_read_case_list(args.whitelist))
    blacklist.extend(_read_case_list(args.blacklist))
    return (whitelist or None, blacklist or None)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _is_port_in_use_failure(output: str) -> bool:
    lowered = output.lower()
    return "eaddrinuse" in lowered or "address already in use" in lowered


def _run_subprocess_command(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    captured_lines: list[str] = []
    for line in proc.stdout:
        print(line, end="")
        captured_lines.append(line)
    returncode = proc.wait()
    return returncode, "".join(captured_lines)


def _init_output_file(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def _append_line(path: Path | None, line: str) -> None:
    if path is None:
        return
    with path.open("a") as handle:
        handle.write(line)
        handle.write("\n")


def _format_case_status(index: int, total: int, result: CaseResult) -> str:
    status = "PASS" if result.ok else "FAIL"
    suffix = f"(module={result.case.module_key}, rc={result.returncode}"
    if result.duration_sec is not None:
        suffix += f", {result.duration_sec:.1f}s"
    suffix += ")"
    return f"[{status} {index}/{total}] {result.case.name} {suffix}"


def _finalize_matrix_results(
    *,
    results: list[CaseResult],
    reported_failed_names: list[str],
    report_path: Path | None,
    mode_label: str,
) -> None:
    failed = [result for result in results if not result.ok]
    failed_names = [result.case.name for result in failed]
    passed = len(results) - len(failed)
    summary = f"Full-stack matrix done: pass={passed} fail={len(failed)} total={len(results)}"
    print(summary)
    _append_line(report_path, summary)
    if reported_failed_names != failed_names:
        mismatch = (
            "Matrix runner internal inconsistency: "
            f"reported_failed_names={reported_failed_names} results_failed_names={failed_names}"
        )
        print(mismatch)
        _append_line(report_path, mismatch)
        raise SystemExit(2)
    if failed_names:
        failed_line = f"Failed cases: {', '.join(failed_names)}"
        print(failed_line)
        _append_line(report_path, failed_line)
        raise SystemExit(1)
    print(f"{mode_label} PASS")


def _group_key(case: MatrixCase) -> str:
    parts = case.name.split("/")
    if len(parts) < 2:
        return case.name
    return f"{parts[0]}/{parts[1]}"


def _group_cases(cases: list[MatrixCase]) -> list[tuple[str, list[MatrixCase]]]:
    groups: list[tuple[str, list[MatrixCase]]] = []
    for case in cases:
        key = _group_key(case)
        if groups and groups[-1][0] == key:
            groups[-1][1].append(case)
        else:
            groups.append((key, [case]))
    return groups


def _append_completed_case(path_str: str | None, case_name: str) -> None:
    if path_str is None or dist.get_rank() != 0:
        return
    with Path(path_str).open("a") as handle:
        handle.write(case_name)
        handle.write("\n")


def _read_completed_cases(path_str: str) -> list[str]:
    path = Path(path_str)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _terminate_process_context(context, *, grace_period_sec: float = 5.0) -> None:
    for process in getattr(context, "processes", []):
        if process.is_alive():
            process.terminate()
    deadline = time.perf_counter() + grace_period_sec
    for process in getattr(context, "processes", []):
        remaining = max(0.0, deadline - time.perf_counter())
        process.join(timeout=remaining)
    for process in getattr(context, "processes", []):
        if process.is_alive():
            process.kill()
        process.join()


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


def _reset_merged_case_state(device: torch.device | None) -> None:
    """Clear process-local test state which a subprocess would discard."""
    # A case creates and destroys several NCCL subgroups.  Their Python handles
    # may be gone while CUDA work queued through those communicators is still
    # retiring.  Fence every rank through the default group before the next
    # case creates fresh subgroups; otherwise a long merged EP/1F1B suite can
    # race communicator teardown with the next case's collectives.
    if device is not None:
        torch.cuda.synchronize(device)
    dist.barrier()
    # RuntimeCore/plugin hooks form reference cycles.  Reclaim them before
    # clearing the process-global test allocator so one case cannot retain or
    # alias the next case's buffers.
    gc.collect()
    clear_buffer_pool()
    if device is not None:
        torch.cuda.empty_cache()
    dist.barrier()


def _run_merged_worker(rank: int, args: argparse.Namespace) -> None:
    case_names, blacklist_names = _resolve_case_names(args)
    device: torch.device | None = None
    if args.backend == "nccl":
        cuda_device_count = torch.cuda.device_count()
        if cuda_device_count <= 0:
            raise RuntimeError("NCCL matrix runner requires at least one visible CUDA device")
        local_rank = rank % cuda_device_count
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
            blacklist_names=blacklist_names,
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
                _append_completed_case(args.completed_cases_file, case.name)
            finally:
                _cleanup_checkpoint_dir(checkpoint_dir)
                _reset_merged_case_state(device)
        if rank == 0:
            print("Merged full-stack matrix PASS")
    finally:
        dist.destroy_process_group()


def _run_subprocess_case(case: MatrixCase) -> None:
    checkpoint_dir: str | None = None
    try:
        if case.needs_checkpoint_dir:
            safe_name = case.name.replace("/", "_")
            checkpoint_dir = tempfile.mkdtemp(prefix=f"{safe_name}_")
        print(f"=== {case.name} ===")
        last_output = ""
        for attempt in range(1, _PORT_RETRY_LIMIT + 1):
            case_args = dict(case.args)
            case_args["master_port"] = _find_free_port()
            cli_args = MatrixCase(
                name=case.name,
                module_key=case.module_key,
                args=case_args,
                needs_checkpoint_dir=case.needs_checkpoint_dir,
            ).to_cli_args()
            if checkpoint_dir is not None:
                cli_args.extend(["--checkpoint-dir", checkpoint_dir])
            cmd = [sys.executable, MODULE_SCRIPTS[case.module_key], *cli_args]
            returncode, output = _run_subprocess_command(cmd)
            if returncode == 0:
                return
            last_output = output
            if attempt < _PORT_RETRY_LIMIT and _is_port_in_use_failure(output):
                print(
                    f"[retry {attempt}/{_PORT_RETRY_LIMIT - 1}] {case.name}: "
                    "TCPStore port busy, retrying with a new port"
                )
                continue
            raise subprocess.CalledProcessError(returncode=returncode, cmd=cmd, output=last_output)
    finally:
        if checkpoint_dir is not None:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


def _run_subprocess_case_keep_going(case: MatrixCase) -> CaseResult:
    checkpoint_dir: str | None = None
    start = time.perf_counter()
    returncode = 0
    try:
        if case.needs_checkpoint_dir:
            safe_name = case.name.replace("/", "_")
            checkpoint_dir = tempfile.mkdtemp(prefix=f"{safe_name}_")
        print(f"=== {case.name} ===")
        for attempt in range(1, _PORT_RETRY_LIMIT + 1):
            case_args = dict(case.args)
            case_args["master_port"] = _find_free_port()
            cli_args = MatrixCase(
                name=case.name,
                module_key=case.module_key,
                args=case_args,
                needs_checkpoint_dir=case.needs_checkpoint_dir,
            ).to_cli_args()
            if checkpoint_dir is not None:
                cli_args.extend(["--checkpoint-dir", checkpoint_dir])
            cmd = [sys.executable, MODULE_SCRIPTS[case.module_key], *cli_args]
            returncode, output = _run_subprocess_command(cmd)
            if returncode == 0:
                break
            if attempt < _PORT_RETRY_LIMIT and _is_port_in_use_failure(output):
                print(
                    f"[retry {attempt}/{_PORT_RETRY_LIMIT - 1}] {case.name}: "
                    "TCPStore port busy, retrying with a new port"
                )
                continue
            break
        ok = returncode == 0
        return CaseResult(
            case=case,
            ok=ok,
            duration_sec=time.perf_counter() - start,
            returncode=returncode,
        )
    finally:
        if checkpoint_dir is not None:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


def run_subprocess_matrix(args: argparse.Namespace) -> None:
    case_names, blacklist_names = _resolve_case_names(args)
    cases = build_full_stack_matrix_cases(
        backend=args.backend,
        world_size=args.world_size,
        case_names=case_names,
        blacklist_names=blacklist_names,
        case_filter=args.case_filter,
        max_cases=args.max_cases,
    )
    print(f"Selected {len(cases)} subprocess full-stack cases")
    report_path = _init_output_file(args.report_file)
    failures_path = _init_output_file(args.failures_file)
    passes_path = _init_output_file(args.passes_file)
    results: list[CaseResult] = []
    reported_failed_names: list[str] = []
    for index, case in enumerate(cases, start=1):
        result = _run_subprocess_case_keep_going(case)
        results.append(result)
        status_line = _format_case_status(index, len(cases), result)
        print(status_line)
        _append_line(report_path, status_line)
        if result.ok:
            _append_line(passes_path, case.name)
        else:
            reported_failed_names.append(case.name)
            _append_line(failures_path, case.name)
    _finalize_matrix_results(
        results=results,
        reported_failed_names=reported_failed_names,
        report_path=report_path,
        mode_label="Subprocess full-stack matrix",
    )


def _run_merged_group_once(args: argparse.Namespace, cases: list[MatrixCase], completed_file: str) -> None:
    group_args = argparse.Namespace(**vars(args))
    group_args.case_names = [case.name for case in cases]
    group_args.blacklist_names = None
    group_args.whitelist = None
    group_args.blacklist = None
    group_args.case_filter = None
    group_args.max_cases = None
    group_args.completed_cases_file = completed_file
    Path(completed_file).write_text("")
    context = mp.start_processes(
        _run_merged_worker,
        args=(group_args,),
        nprocs=args.world_size,
        join=False,
        start_method="spawn",
    )
    last_completed_count = 0
    last_progress_at = time.perf_counter()
    timeout_sec = args.case_timeout_sec
    try:
        while True:
            completed_names = _read_completed_cases(completed_file)
            completed_count = len(completed_names)
            if completed_count != last_completed_count:
                last_completed_count = completed_count
                last_progress_at = time.perf_counter()
            try:
                if context.join(timeout=1.0, grace_period=5.0):
                    return
            except BaseException:
                _terminate_process_context(context)
                raise
            if timeout_sec is not None and timeout_sec > 0:
                stalled_sec = time.perf_counter() - last_progress_at
                if stalled_sec > timeout_sec and completed_count < len(cases):
                    case_name = cases[completed_count].name
                    _terminate_process_context(context)
                    raise MergedCaseTimeoutError(case_name, timeout_sec)
    finally:
        _terminate_process_context(context)


def run_grouped_merged_matrix(args: argparse.Namespace) -> None:
    case_names, blacklist_names = _resolve_case_names(args)
    cases = build_full_stack_matrix_cases(
        backend=args.backend,
        world_size=args.world_size,
        case_names=case_names,
        blacklist_names=blacklist_names,
        case_filter=args.case_filter,
        max_cases=args.max_cases,
    )
    print(f"Selected {len(cases)} merged full-stack cases")
    report_path = _init_output_file(args.report_file)
    failures_path = _init_output_file(args.failures_file)
    passes_path = _init_output_file(args.passes_file)
    results: list[CaseResult] = []
    reported_failed_names: list[str] = []
    total = len(cases)
    next_index = 1

    for group_key, group_cases in _group_cases(cases):
        print(f"=== group {group_key} ({len(group_cases)} cases) ===")
        remaining = list(group_cases)
        while remaining:
            with tempfile.NamedTemporaryFile(prefix="maltos_completed_", suffix=".txt", delete=False) as handle:
                completed_file = handle.name
            start = time.perf_counter()
            try:
                _run_merged_group_once(args, remaining, completed_file)
                duration_sec = time.perf_counter() - start
                for case in remaining:
                    result = CaseResult(case=case, ok=True, duration_sec=duration_sec, returncode=0)
                    results.append(result)
                    status_line = _format_case_status(next_index, total, result)
                    print(status_line)
                    _append_line(report_path, status_line)
                    _append_line(passes_path, case.name)
                    next_index += 1
                remaining = []
            except BaseException as exc:
                # The merged worker owns a single process group, so an error from
                # one rank is surfaced here as ProcessRaisedException.  Preserve
                # its nested worker traceback; otherwise every failure looks like
                # an opaque rc=1 in the matrix report.
                traceback.print_exc()
                duration_sec = time.perf_counter() - start
                completed_names = _read_completed_cases(completed_file)
                completed_count = len(completed_names)
                for case in remaining[:completed_count]:
                    result = CaseResult(case=case, ok=True, duration_sec=duration_sec, returncode=0)
                    results.append(result)
                    status_line = _format_case_status(next_index, total, result)
                    print(status_line)
                    _append_line(report_path, status_line)
                    _append_line(passes_path, case.name)
                    next_index += 1
                if completed_count >= len(remaining):
                    raise
                failed_case = remaining[completed_count]
                returncode = 124 if isinstance(exc, MergedCaseTimeoutError) else 1
                if isinstance(exc, MergedCaseTimeoutError):
                    hang_line = f"[HANG] {failed_case.name} exceeded {exc.timeout_sec:.0f}s without completion"
                    print(hang_line)
                    _append_line(report_path, hang_line)
                result = CaseResult(case=failed_case, ok=False, duration_sec=duration_sec, returncode=returncode)
                results.append(result)
                status_line = _format_case_status(next_index, total, result)
                print(status_line)
                _append_line(report_path, status_line)
                reported_failed_names.append(failed_case.name)
                _append_line(failures_path, failed_case.name)
                next_index += 1
                remaining = remaining[completed_count + 1 :]
            finally:
                Path(completed_file).unlink(missing_ok=True)
    _finalize_matrix_results(
        results=results,
        reported_failed_names=reported_failed_names,
        report_path=report_path,
        mode_label="Merged full-stack matrix",
    )


def _print_case_names(args: argparse.Namespace) -> None:
    case_names, blacklist_names = _resolve_case_names(args)
    cases = build_full_stack_matrix_cases(
        backend=args.backend,
        world_size=args.world_size,
        case_names=case_names,
        blacklist_names=blacklist_names,
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
    if args.merge is True or (args.merge is None and args.backend == "nccl"):
        run_grouped_merged_matrix(args)
    else:
        run_subprocess_matrix(args)


if __name__ == "__main__":
    main()
