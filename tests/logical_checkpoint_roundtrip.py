from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distributed_test_utils import max_diff
from models import TinyModel, TinyTransformerTp
from parallel import ParallelPlan
from runtime import MeshConfig, RuntimeCore
from runtime.plugins.tp import TensorParallelPlugin
from runtime.plugins.zero3 import Zero3Plugin
from state import (
    iter_logical_tensors_from_runtime_checkpoint,
    iter_logical_checkpoint_tensors,
    load_logical_checkpoint,
    load_logical_tensor,
    save_logical_checkpoint,
    save_sharded_checkpoint,
)


_TP_MODEL_KWARGS = dict(
    dim=64,
    n_heads=4,
    n_kv_heads=4,
    hidden_size=128,
    eps=1e-5,
    n_layers=1,
    vocab_size=128,
    max_seq_len=32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29562)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--mode", type=str, default="all", choices=("all", "tp", "zero3"))
    return parser.parse_args()


def _build_tp_core(seed: int, world_size: int) -> RuntimeCore:
    torch.manual_seed(seed)
    model = TinyTransformerTp(**_TP_MODEL_KWARGS)
    core = RuntimeCore(
        mesh=MeshConfig(dp=1, tp=world_size, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.0),
        plugins=[TensorParallelPlugin()],
    )
    core.setup()
    return core


def _build_zero3_core(seed: int, world_size: int) -> tuple[RuntimeCore, Zero3Plugin]:
    torch.manual_seed(seed)
    model = TinyModel(hidden_size=32)
    zero3 = Zero3Plugin()
    core = RuntimeCore(
        mesh=MeshConfig(dp=world_size, tp=1, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.0),
        plugins=[zero3],
    )
    core.setup()
    return core, zero3


def _assert_roundtrip_equal(lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor], *, atol: float, label: str) -> None:
    worst_name, worst = max_diff(lhs, rhs)
    if worst > atol:
        raise AssertionError(f"{label} logical checkpoint roundtrip mismatch: {worst_name} diff={worst:.2e} atol={atol:.2e}")


def _collect_tp_runtime_logical(runtime_dir: str, world_size: int) -> dict[str, torch.Tensor]:
    model = TinyTransformerTp(**_TP_MODEL_KWARGS)
    return dict(
        iter_logical_tensors_from_runtime_checkpoint(
            runtime_dir,
            model=model,
            dp_size=1,
            tp_size=world_size,
            pp_size=1,
            cp_size=1,
            ep_size=1,
        )
    )


def _collect_zero3_runtime_logical(runtime_dir: str, world_size: int) -> dict[str, torch.Tensor]:
    model = TinyModel(hidden_size=32)
    return dict(
        iter_logical_tensors_from_runtime_checkpoint(
            runtime_dir,
            model=model,
            dp_size=world_size,
            tp_size=1,
            pp_size=1,
            cp_size=1,
            ep_size=1,
        )
    )


def _run_tp_case(rank: int, args: argparse.Namespace) -> None:
    source = _build_tp_core(seed=1234, world_size=args.world_size)
    source_runtime_dir = _broadcast_path(tempfile.mkdtemp(prefix="logical_tp_runtime_src_") if rank == 0 else None)
    logical_dir = _broadcast_path(tempfile.mkdtemp(prefix="logical_tp_ckpt_") if rank == 0 else None)

    save_sharded_checkpoint(source.state_manager, source_runtime_dir)

    if rank == 0:
        source_logical = _collect_tp_runtime_logical(source_runtime_dir, args.world_size)
        assert "rope.cos" in source_logical
        assert "rope.sin" in source_logical
        save_logical_checkpoint(logical_dir, source_logical, max_shard_size_bytes=4096)
        loaded = load_logical_checkpoint(logical_dir)
        iterated = dict(iter_logical_checkpoint_tensors(logical_dir))
        rope_cos = load_logical_tensor(logical_dir, "rope.cos")
        shard_files = sorted(Path(logical_dir).glob("model-*.safetensors"))
    else:
        source_logical = None
        loaded = None
        iterated = None
        rope_cos = None
        shard_files = None
    loaded = _broadcast_tensor_dict(loaded)
    iterated = _broadcast_tensor_dict(iterated)
    rope_cos = _broadcast_tensor(rope_cos)
    shard_files = _broadcast_string_list([str(path) for path in shard_files] if shard_files is not None else None)
    if rank == 0:
        assert source_logical is not None
        if len(shard_files) < 2:
            raise AssertionError(f"expected multi-shard logical checkpoint, got {len(shard_files)} shard(s)")
        _assert_roundtrip_equal(source_logical, loaded, atol=1e-6, label="tp load")
        _assert_roundtrip_equal(source_logical, iterated, atol=1e-6, label="tp iter")
        if not torch.equal(rope_cos, source_logical["rope.cos"]):
            raise AssertionError("tp logical checkpoint single-tensor load mismatch for rope.cos")


def _run_zero3_case(rank: int, args: argparse.Namespace) -> None:
    source, _ = _build_zero3_core(seed=1234, world_size=args.world_size)
    source_runtime_dir = _broadcast_path(tempfile.mkdtemp(prefix="logical_zero3_runtime_src_") if rank == 0 else None)
    logical_dir = _broadcast_path(tempfile.mkdtemp(prefix="logical_zero3_ckpt_") if rank == 0 else None)
    save_sharded_checkpoint(source.state_manager, source_runtime_dir)

    if rank == 0:
        source_logical = _collect_zero3_runtime_logical(source_runtime_dir, args.world_size)
        save_logical_checkpoint(logical_dir, source_logical)
        logical_from_runtime = load_logical_checkpoint(logical_dir)
        iterated = dict(iter_logical_checkpoint_tensors(logical_dir))
        sample_name = sorted(source_logical)[0]
        sample_tensor = load_logical_tensor(logical_dir, sample_name)
    else:
        source_logical = None
        logical_from_runtime = None
        iterated = None
        sample_name = None
        sample_tensor = None
    logical_from_runtime = _broadcast_tensor_dict(logical_from_runtime)
    iterated = _broadcast_tensor_dict(iterated)
    sample_name = _broadcast_string(sample_name)
    sample_tensor = _broadcast_tensor(sample_tensor)
    if rank == 0:
        assert source_logical is not None
        assert sample_name is not None
        _assert_roundtrip_equal(source_logical, logical_from_runtime, atol=1e-6, label="zero3 load")
        _assert_roundtrip_equal(source_logical, iterated, atol=1e-6, label="zero3 iter")
        if not torch.equal(sample_tensor, source_logical[sample_name]):
            raise AssertionError(f"zero3 logical checkpoint single-tensor load mismatch for {sample_name}")


def _broadcast_path(path: str | None) -> str:
    if not dist.is_initialized():
        assert path is not None
        return path
    payload = [path]
    dist.broadcast_object_list(payload, src=0)
    assert payload[0] is not None
    return str(payload[0])


def _broadcast_tensor_dict(payload: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
    if not dist.is_initialized():
        assert payload is not None
        return payload
    objects = [payload]
    dist.broadcast_object_list(objects, src=0)
    assert objects[0] is not None
    return objects[0]


def _broadcast_tensor(payload: torch.Tensor | None) -> torch.Tensor:
    if not dist.is_initialized():
        assert payload is not None
        return payload
    objects = [payload]
    dist.broadcast_object_list(objects, src=0)
    assert objects[0] is not None
    return objects[0]


def _broadcast_string(payload: str | None) -> str:
    if not dist.is_initialized():
        assert payload is not None
        return payload
    objects = [payload]
    dist.broadcast_object_list(objects, src=0)
    assert objects[0] is not None
    return str(objects[0])


def _broadcast_string_list(payload: list[str] | None) -> list[str]:
    if not dist.is_initialized():
        assert payload is not None
        return payload
    objects = [payload]
    dist.broadcast_object_list(objects, src=0)
    assert objects[0] is not None
    return [str(item) for item in objects[0]]


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )
    if args.mode in {"all", "tp"}:
        _run_tp_case(rank, args)
    if args.mode in {"all", "zero3"}:
        _run_zero3_case(rank, args)
    if rank == 0:
        print("PASS")
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
