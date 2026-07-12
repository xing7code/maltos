"""Resume test for TP+SP+ZeRO3+bf16+grad-clip RuntimeCore checkpointing."""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distributed_test_utils import (
    max_diff as _max_diff,
    named_tensors as _named_tensors,
    normalize_param_name as _normalize_param_name,
    rule_by_param_name as _rule_by_param_name,
    supports_bf16_autocast as _supports_bf16_autocast,
)
from models import TinyTransformer, TinyTransformerTpSp
from models.tiny_transformer import RmsNorm
from helpers import causal_lm_batch
from parallel import ParallelPlan
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.plugins.precision import PrecisionPlugin
from runtime.plugins.sp import SequenceParallelPlugin
from runtime.plugins.tp import TensorParallelPlugin
from runtime.plugins.zero3 import Zero3Plugin
from state import load_sharded_checkpoint, save_sharded_checkpoint


_MODEL_KWARGS = dict(
    dim=64,
    n_heads=4,
    n_kv_heads=4,
    hidden_size=128,
    eps=1e-5,
    n_layers=2,
    vocab_size=256,
    max_seq_len=64,
)
_LOSS_ATOL = 5e-2
_STEP_ATOL = 5e-2
_LR = 1e-2
_ZERO3_WRAP_CLS = {torch.nn.Linear, torch.nn.Embedding, RmsNorm}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29536)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def _mesh_indices(rank: int, tp_size: int) -> tuple[int, int]:
    return rank // tp_size, rank % tp_size


def _build_reference(seed: int, batch_size: int, seq_len: int) -> tuple[TinyTransformer, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    first_tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len))
    second_tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len))
    model = TinyTransformer(**_MODEL_KWARGS)
    return model, first_tokens, second_tokens


def _local_tokens(full_tokens: torch.Tensor, dp_idx: int, dp_size: int) -> torch.Tensor:
    local_batch_size = full_tokens.size(0) // dp_size
    return full_tokens.narrow(0, dp_idx * local_batch_size, local_batch_size).contiguous()

def _build_runtime(model: TinyTransformerTpSp, dp_size: int, tp_size: int) -> tuple[RuntimeCore, Zero3Plugin]:
    zero3 = Zero3Plugin(
        wrap_cls=_ZERO3_WRAP_CLS,
    )
    core = RuntimeCore(
        mesh=MeshConfig(dp=dp_size, tp=tp_size, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=model,
        grad_clip_max_norm=1.0,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=[
            TensorParallelPlugin(),
            SequenceParallelPlugin(),
            zero3,
            PrecisionPlugin(compute_dtype=torch.bfloat16),
        ],
    )
    core.setup()
    return core, zero3


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )
    if args.world_size != args.dp_size * args.tp_size:
        raise ValueError("world size must equal dp_size * tp_size")
    if args.global_batch_size % args.dp_size != 0:
        raise ValueError("global batch size must be divisible by dp size")

    if not _supports_bf16_autocast():
        if rank == 0:
            print("SKIP: bf16 autocast is not supported on this runtime")
        dist.destroy_process_group()
        return

    dp_idx, _ = _mesh_indices(rank, args.tp_size)
    baseline_model, first_tokens, second_tokens = _build_reference(args.seed, args.global_batch_size, args.seq_len)

    continuous_model = TinyTransformerTpSp(**_MODEL_KWARGS)
    continuous_model.load_state_dict(baseline_model.state_dict())
    continuous_core, continuous_zero3 = _build_runtime(continuous_model, args.dp_size, args.tp_size)

    restored_model = TinyTransformerTpSp(**_MODEL_KWARGS)
    restored_model.load_state_dict(baseline_model.state_dict())
    restored_core, restored_zero3 = _build_runtime(restored_model, args.dp_size, args.tp_size)

    first_local = _local_tokens(first_tokens, dp_idx, args.dp_size)
    second_local = _local_tokens(second_tokens, dp_idx, args.dp_size)

    _, should_step = continuous_core.run_step(causal_lm_batch(first_local))
    continuous_core.step_optimizer()
    save_sharded_checkpoint(continuous_core.state_manager, args.checkpoint_dir)
    continuous_second_loss, _ = continuous_core.run_step(causal_lm_batch(second_local))
    continuous_core.step_optimizer()

    load_sharded_checkpoint(restored_core.state_manager, args.checkpoint_dir)
    restored_second_loss, _ = restored_core.run_step(causal_lm_batch(second_local))
    restored_core.step_optimizer()

    dp_group = continuous_core.get_group(MeshAxis.DP)
    continuous_loss = continuous_second_loss.detach().clone()
    restored_loss = restored_second_loss.detach().clone()
    if dp_group is not None:
        dist.all_reduce(continuous_loss, op=dist.ReduceOp.AVG, group=dp_group)
        dist.all_reduce(restored_loss, op=dist.ReduceOp.AVG, group=dp_group)

    continuous_zero3.materialize_model()
    restored_zero3.materialize_model()
    shard_rules = _rule_by_param_name(continuous_model)
    tp_group = continuous_core.get_group(MeshAxis.TP)
    continuous_params = _named_tensors(continuous_core.model, shard_rules, tp_group, normalize_name=_normalize_param_name)
    restored_params = _named_tensors(restored_core.model, shard_rules, tp_group, normalize_name=_normalize_param_name)
    param_name, param_diff = _max_diff(continuous_params, restored_params)

    if rank == 0:
        loss_diff = abs(continuous_loss.item() - restored_loss.item())
        print(f"Checkpoint dir    : {args.checkpoint_dir}")
        print(f"Resume loss diff  : {loss_diff:.2e}  (atol={_LOSS_ATOL:.2e})")
        print(f"Resume param diff : {param_diff:.2e}  ({param_name}, atol={_STEP_ATOL:.2e})")
        if loss_diff > _LOSS_ATOL:
            raise AssertionError(f"loss mismatch after resume: diff={loss_diff:.2e}")
        if param_diff > _STEP_ATOL:
            raise AssertionError(f"param mismatch after resume: {param_name} diff={param_diff:.2e}")
        if continuous_core.state.metadata.get("loss_scale") is not None:
            raise AssertionError("bf16 path should keep loss_scale=None")
        if restored_core.state.metadata.get("loss_scale") is not None:
            raise AssertionError("bf16 resume path should keep loss_scale=None")
        if continuous_core.state.metadata.get("overflow") is not False:
            raise AssertionError("bf16 path should keep overflow=False")
        if restored_core.state.metadata.get("overflow") is not False:
            raise AssertionError("bf16 resume path should keep overflow=False")
        print("PASS")

    continuous_zero3.reshard_model()
    restored_zero3.reshard_model()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = tempfile.mkdtemp(prefix="tp_sp_zero3_bf16_clip_resume_")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
