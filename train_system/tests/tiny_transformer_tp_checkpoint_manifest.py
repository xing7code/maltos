"""Manifest test for TP checkpoint annotations."""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train_system.models import TinyTransformerTp
from train_system.parallel import ParallelPlan
from train_system.runtime import MeshConfig, RuntimeCore
from train_system.runtime.plugins.tp import TensorParallelPlugin
from train_system.state import save_sharded_checkpoint


_MODEL_KWARGS = dict(
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
    parser.add_argument("--master-port", type=int, default=29531)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )

    torch.manual_seed(1234)
    model = TinyTransformerTp(**_MODEL_KWARGS)
    core = RuntimeCore(
        mesh=MeshConfig(dp=1, tp=args.world_size, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        plugins=[TensorParallelPlugin()],
    )
    core.setup()
    save_sharded_checkpoint(core.state_manager, args.checkpoint_dir)

    if rank == 0:
        with open(os.path.join(args.checkpoint_dir, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        entries = {
            entry["state_key"]: entry
            for entry in manifest["ranks"][0]["entries"]
        }
        q_proj = entries["layers.0.attn.q_proj.weight"]
        o_proj = entries["layers.0.attn.o_proj.weight"]
        assert q_proj["physical_shape"] == [32, 64]
        assert q_proj["logical_shapes"] == [[64, 64]]
        assert q_proj["annotations"]["tp"]["shards"][0]["axis"] == "param_out"
        assert o_proj["physical_shape"] == [64, 32]
        assert o_proj["logical_shapes"] == [[64, 64]]
        assert o_proj["annotations"]["tp"]["shards"][0]["axis"] == "param_in"
        print("PASS")

    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = tempfile.mkdtemp(prefix="tp_ckpt_")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
