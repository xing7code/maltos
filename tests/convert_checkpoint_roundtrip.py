from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distributed_test_utils import max_diff
from state import load_logical_checkpoint, save_runtime_spec, save_sharded_checkpoint
from train.cli import _build_model, _build_runtime
from train.flags import build_runtime_spec, parse_args_from


@dataclass(frozen=True)
class Case:
    name: str
    world_size: int
    model: str
    dp_size: int
    pp_size: int
    tp_size: int
    ep_size: int
    zero_stage: int
    num_experts: int = 8


_CASES = {
    "dense": Case(name="dense", world_size=1, model="tiny", dp_size=1, pp_size=1, tp_size=1, ep_size=1, zero_stage=0),
    "tp": Case(name="tp", world_size=2, model="tiny", dp_size=1, pp_size=1, tp_size=2, ep_size=1, zero_stage=0),
    "pp": Case(name="pp", world_size=2, model="tiny", dp_size=1, pp_size=2, tp_size=1, ep_size=1, zero_stage=0),
    "zero3": Case(name="zero3", world_size=2, model="tiny", dp_size=2, pp_size=1, tp_size=1, ep_size=1, zero_stage=3),
    "tp_zero3": Case(name="tp_zero3", world_size=4, model="tiny", dp_size=2, pp_size=1, tp_size=2, ep_size=1, zero_stage=3),
    "ep": Case(name="ep", world_size=2, model="tiny_moe", dp_size=2, pp_size=1, tp_size=1, ep_size=2, zero_stage=0, num_experts=4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all", choices=("all", *sorted(_CASES)))
    parser.add_argument("--backend", type=str, default="gloo")
    return parser.parse_args()


def _recipe_argv(case: Case) -> list[str]:
    return [
        "--model",
        case.model,
        "--dim",
        "64",
        "--n-heads",
        "4",
        "--n-kv-heads",
        "4",
        "--hidden-size",
        "128",
        "--n-layers",
        "1",
        "--vocab-size",
        "128",
        "--seq-len",
        "16",
        "--max-steps",
        "1",
        "--micro-batch-size",
        "1",
        "--dp-size",
        str(case.dp_size),
        "--pp-size",
        str(case.pp_size),
        "--tp-size",
        str(case.tp_size),
        "--ep-size",
        str(case.ep_size),
        "--zero-stage",
        str(case.zero_stage),
        "--num-experts",
        str(case.num_experts),
        "--precision",
        "fp32",
    ]


def _build_recipe_args(case: Case):
    return parse_args_from(_recipe_argv(case), require_data=False)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _assert_logical_equal(lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor], *, label: str) -> None:
    worst_name, worst = max_diff(lhs, rhs)
    if worst > 1e-6:
        raise AssertionError(f"{label} logical mismatch: {worst_name} diff={worst:.2e}")


def _create_runtime_checkpoint_worker(rank: int, case: Case, backend: str, master_port: int, checkpoint_root: str) -> None:
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=case.world_size,
    )
    args = _build_recipe_args(case)
    torch.manual_seed(args.seed)
    model = _build_model(args)
    runtime = _build_runtime(args, model, torch.device("cpu"), weights_only=True)
    checkpoint_root_path = Path(checkpoint_root)
    checkpoint_dir = checkpoint_root_path / "step_00000001"
    runtime.setup()
    try:
        if rank == 0:
            save_runtime_spec(checkpoint_root_path, build_runtime_spec(args))
        dist.barrier()
        save_sharded_checkpoint(runtime.state_manager, checkpoint_dir)
    finally:
        runtime.close()
        dist.barrier()
        dist.destroy_process_group()


def _logical_to_runtime_worker(
    rank: int,
    case: Case,
    backend: str,
    master_port: int,
    logical_dir: str,
    output_root: str,
    config_path: str,
) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": ".",
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(case.world_size),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(master_port),
    }
    cmd = [
        sys.executable,
        "tools/convert_checkpoint.py",
        "logical-to-runtime",
        "--config",
        config_path,
        "--checkpoint",
        logical_dir,
        "--output",
        output_root,
        "--",
        *_recipe_argv(case),
        "--backend",
        backend,
        "--master-addr",
        "127.0.0.1",
        "--master-port",
        str(master_port),
    ]
    subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
    )


def _run_case(case: Case, *, backend: str) -> None:
    with tempfile.TemporaryDirectory(prefix=f"convert_roundtrip_{case.name}_") as tmp:
        root = Path(tmp)
        source_runtime_root = root / "source_runtime"
        logical_dir = root / "logical"
        restored_runtime_root = root / "restored_runtime"
        restored_logical_dir = root / "restored_logical"
        config_path = root / "empty.yaml"
        config_path.write_text("{}\n", encoding="utf-8")

        create_port = _free_port()
        mp.spawn(
            _create_runtime_checkpoint_worker,
            args=(case, backend, create_port, str(source_runtime_root)),
            nprocs=case.world_size,
            join=True,
        )

        subprocess.run(
            [
                sys.executable,
                "tools/convert_checkpoint.py",
                "runtime-to-logical",
                "--checkpoint",
                source_runtime_root / "step_00000001",
                "--output",
                logical_dir,
            ],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONPATH": "."},
            check=True,
        )
        if not (logical_dir / "model.safetensors.index.json").is_file():
            raise AssertionError(f"{case.name}: expected logical checkpoint index")

        convert_port = _free_port()
        mp.spawn(
            _logical_to_runtime_worker,
            args=(case, backend, convert_port, str(logical_dir), str(restored_runtime_root), str(config_path)),
            nprocs=case.world_size,
            join=True,
        )
        restored_manifest = json.loads((restored_runtime_root / "step_00000000" / "manifest.json").read_text(encoding="utf-8"))
        artifact_kinds = {artifact["kind"] for artifact in restored_manifest["artifacts"]}
        if artifact_kinds != {"model"}:
            raise AssertionError(f"{case.name}: logical-to-runtime should emit weights-only runtime checkpoint, got artifacts={sorted(artifact_kinds)}")

        subprocess.run(
            [
                sys.executable,
                "tools/convert_checkpoint.py",
                "runtime-to-logical",
                "--checkpoint",
                restored_runtime_root / "step_00000000",
                "--output",
                restored_logical_dir,
            ],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONPATH": "."},
            check=True,
        )
        expected = load_logical_checkpoint(logical_dir)
        restored = load_logical_checkpoint(restored_logical_dir)
        _assert_logical_equal(expected, restored, label=case.name)


def main() -> None:
    args = parse_args()
    case_names = sorted(_CASES) if args.mode == "all" else [args.mode]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for case_name in case_names:
        _run_case(_CASES[case_name], backend=args.backend)
    print("PASS")


if __name__ == "__main__":
    main()
