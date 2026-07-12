"""Resume test for EP+ZeRO3 RuntimeCore checkpointing."""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from helpers import causal_lm_batch
from models import TinyMoETransformer
from models.tiny_transformer import RmsNorm
from parallel import ParallelPlan
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.plugins.ep import ExpertParallelPlugin
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
    num_experts=4,
)
_LOSS_ATOL = 5e-5
_STEP_ATOL = 5e-5
_LR = 1e-2
_ZERO3_WRAP_CLS = {torch.nn.Linear, torch.nn.Embedding, RmsNorm}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29610)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def _build_reference(seed: int, batch_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    first_tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len))
    second_tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len))
    return first_tokens, second_tokens


def _local_tokens(full_tokens: torch.Tensor, rank: int, dp_size: int) -> torch.Tensor:
    local_batch_size = full_tokens.size(0) // dp_size
    return full_tokens.narrow(0, rank * local_batch_size, local_batch_size).contiguous()


def _build_runtime(seed: int, dp_size: int, ep_size: int) -> tuple[RuntimeCore, Zero3Plugin]:
    torch.manual_seed(seed)
    model = TinyMoETransformer(**_MODEL_KWARGS)
    zero3 = Zero3Plugin(wrap_cls=_ZERO3_WRAP_CLS)
    core = RuntimeCore(
        mesh=MeshConfig(dp=dp_size, tp=1, pp=1, cp=1, ep=ep_size),
        plan=ParallelPlan(),
        model=model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=[ExpertParallelPlugin(), zero3],
    )
    core.setup()
    return core, zero3


def _local_max_param_diff(lhs: torch.nn.Module, rhs: torch.nn.Module) -> tuple[str, float]:
    worst_name = ""
    worst_diff = 0.0
    for (lhs_name, lhs_param), (rhs_name, rhs_param) in zip(lhs.named_parameters(), rhs.named_parameters()):
        assert lhs_name == rhs_name
        diff = (lhs_param.detach() - rhs_param.detach()).abs().max().item()
        if diff > worst_diff:
            worst_name = lhs_name
            worst_diff = diff
    return worst_name, worst_diff


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )
    if args.world_size != args.dp_size:
        raise ValueError("EP resume test expects world_size == dp_size")
    if args.dp_size < args.ep_size:
        raise ValueError("EP resume test expects dp_size >= ep_size")
    if args.global_batch_size % args.dp_size != 0:
        raise ValueError("global batch size must be divisible by dp size")

    first_tokens, second_tokens = _build_reference(args.seed, args.global_batch_size, args.seq_len)
    first_local = _local_tokens(first_tokens, rank, args.dp_size)
    second_local = _local_tokens(second_tokens, rank, args.dp_size)

    continuous_core, continuous_zero3 = _build_runtime(args.seed, args.dp_size, args.ep_size)
    _, should_step = continuous_core.run_step(causal_lm_batch(first_local))
    if not should_step:
        raise AssertionError("continuous EP+ZeRO3 first step should step optimizer")
    continuous_core.step_optimizer()
    dist.barrier()
    save_sharded_checkpoint(continuous_core.state_manager, args.checkpoint_dir)
    dist.barrier()
    continuous_second_loss, should_step = continuous_core.run_step(causal_lm_batch(second_local))
    if not should_step:
        raise AssertionError("continuous EP+ZeRO3 second step should step optimizer")
    continuous_core.step_optimizer()
    dist.barrier()

    restored_core, restored_zero3 = _build_runtime(args.seed, args.dp_size, args.ep_size)
    load_sharded_checkpoint(restored_core.state_manager, args.checkpoint_dir)
    dist.barrier()
    restored_second_loss, should_step = restored_core.run_step(causal_lm_batch(second_local))
    if not should_step:
        raise AssertionError("restored EP+ZeRO3 second step should step optimizer")
    restored_core.step_optimizer()
    dist.barrier()

    reduced_continuous_loss = continuous_second_loss.detach().clone()
    reduced_restored_loss = restored_second_loss.detach().clone()
    dp_group = continuous_core.get_group(MeshAxis.DP)
    if dp_group is not None:
        dist.all_reduce(reduced_continuous_loss, op=dist.ReduceOp.AVG, group=dp_group)
        dist.all_reduce(reduced_restored_loss, op=dist.ReduceOp.AVG, group=dp_group)

    continuous_zero3.materialize_model()
    restored_zero3.materialize_model()
    param_name, local_param_diff = _local_max_param_diff(continuous_core.model, restored_core.model)
    param_diff = torch.tensor(local_param_diff, dtype=torch.float64)
    dist.all_reduce(param_diff, op=dist.ReduceOp.MAX)

    if rank == 0:
        loss_diff = abs(reduced_continuous_loss.item() - reduced_restored_loss.item())
        print(f"Checkpoint dir    : {args.checkpoint_dir}")
        print(f"Resume loss diff  : {loss_diff:.2e}  (atol={_LOSS_ATOL:.2e})")
        print(f"Resume param diff : {param_diff.item():.2e}  ({param_name}, atol={_STEP_ATOL:.2e})")
        if loss_diff > _LOSS_ATOL:
            raise AssertionError(f"EP+ZeRO3 loss mismatch after resume: diff={loss_diff:.2e}")
        if param_diff.item() > _STEP_ATOL:
            raise AssertionError(f"EP+ZeRO3 param mismatch after resume: {param_name} diff={param_diff.item():.2e}")
        print("PASS")

    continuous_zero3.reshard_model()
    restored_zero3.reshard_model()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = tempfile.mkdtemp(prefix="ep_zero3_resume_")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
