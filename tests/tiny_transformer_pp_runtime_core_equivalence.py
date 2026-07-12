"""Equivalence test: baseline TinyTransformer vs RuntimeCore PP variants.

Usage:
  PYTHONPATH=. .venv/bin/python tests/tiny_transformer_pp_runtime_core_equivalence.py \
    --case pp \
    --world-size 2 \
    --pp-size 2 \
    --pp-microbatches 2
"""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distributed_test_utils import (
    max_diff as _max_diff,
    named_tensors as _named_tensors,
    rule_by_param_name as _rule_by_param_name,
)
from helpers import causal_lm_batch
from models import TinyTransformer, TinyTransformerTp, TinyTransformerTpSp
from models.tiny_transformer import RmsNorm
from parallel import ParallelPlan
from parallel.plan import PipelineScheduleConfig
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.plugins.cp import ContextParallelPlugin
from runtime.plugins.ddp import DataParallelPlugin
from runtime.plugins.pp import PipelineParallelPlugin
from runtime.plugins.sp import SequenceParallelPlugin
from runtime.plugins.tp import TensorParallelPlugin
from runtime.plugins.zero1 import Zero1Plugin
from runtime.plugins.zero2 import Zero2Plugin
from runtime.plugins.zero3 import Zero3Plugin


_MODEL_KWARGS = dict(
    dim=64,
    n_heads=4,
    n_kv_heads=4,
    hidden_size=128,
    eps=1e-5,
    n_layers=4,
    vocab_size=256,
    max_seq_len=64,
)

_LOSS_ATOL = 1e-3
_STEP_ATOL = 1e-5
_LR = 1e-2


def _step_atol(args: argparse.Namespace) -> float:
    if args.cp_size > 1:
        return 1e-4
    return _STEP_ATOL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("pp", "pp_ddp_sync", "pp_zero1", "pp_zero2", "pp_zero3", "pp_tp", "pp_tp_sp", "cp_pp"),
        default="pp",
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-microbatches", type=int, default=2)
    parser.add_argument("--pp-schedule", choices=("afab", "1f1b"), default="afab")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29556)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def _build_reference(seed: int, batch_size: int, seq_len: int) -> tuple[TinyTransformer, torch.Tensor]:
    torch.manual_seed(seed)
    tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len))
    model = TinyTransformer(**_MODEL_KWARGS)
    return model, tokens


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )
    if args.world_size != args.dp_size * args.pp_size * args.cp_size * args.tp_size:
        raise ValueError("PP equivalence test expects world_size == dp_size * pp_size * cp_size * tp_size")

    baseline_loss, baseline_params = _run_baseline(rank, args)

    _baseline_model, tokens = _build_reference(args.seed, args.batch_size, args.seq_len)
    pp_model = _build_runtime_model(args.case)
    pp_model.load_state_dict(_baseline_model.state_dict())

    if args.batch_size % args.dp_size != 0:
        raise ValueError("batch_size must be divisible by dp_size")
    dp_idx = rank // (args.pp_size * args.cp_size * args.tp_size)
    local_batch_size = args.batch_size // args.dp_size
    local_tokens = tokens.narrow(0, dp_idx * local_batch_size, local_batch_size).contiguous()

    plugins, zero3 = _make_plugins(args)
    core = RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=args.pp_size, cp=args.cp_size, ep=1),
        plan=ParallelPlan(pp_schedule=PipelineScheduleConfig(microbatches=args.pp_microbatches)),
        model=pp_model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=plugins,
    )
    core.setup()
    runtime_loss, should_step = core.run_step(causal_lm_batch(local_tokens))
    if not should_step:
        raise AssertionError("PP v0 test expects grad_accum_steps=1, should_step must be True")
    core.step_optimizer()

    if zero3 is not None:
        zero3.materialize_model()

    tp_group = core.get_group(MeshAxis.TP)
    shard_rules = _rule_by_param_name(core.model)
    logical_params = _named_tensors(core.model, shard_rules, tp_group)

    local_param_diff = 0.0
    local_param_name = ""
    for name, param in logical_params.items():
        diff = (param - baseline_params[name]).abs().max().item()
        if diff > local_param_diff:
            local_param_diff = diff
            local_param_name = name

    dp_group = core.get_group(MeshAxis.DP)
    cp_group = core.get_group(MeshAxis.CP)
    avg_loss = runtime_loss.detach().clone()
    if cp_group is not None:
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM, group=cp_group)
    if dp_group is not None:
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG, group=dp_group)

    loss_diff_tensor = torch.tensor(abs(avg_loss.item() - baseline_loss), dtype=torch.float64)
    param_diff_tensor = torch.tensor(local_param_diff, dtype=torch.float64)
    dist.all_reduce(loss_diff_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(param_diff_tensor, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"Case             : {args.case}")
        print(f"PP schedule      : {args.pp_schedule}")
        print(f"Baseline loss    : {baseline_loss:.6f}")
        print(f"RuntimeCore PP   : {avg_loss.item():.6f}")
        print(f"Loss diff        : {loss_diff_tensor.item():.2e}  (atol={_LOSS_ATOL:.2e})")
        print(f"Local param diff : {param_diff_tensor.item():.2e}  ({local_param_name}, atol={_step_atol(args):.2e})")
        if loss_diff_tensor.item() > _LOSS_ATOL:
            raise AssertionError(f"PP loss equivalence failed: diff={loss_diff_tensor.item():.2e}")
        if param_diff_tensor.item() > _step_atol(args):
            raise AssertionError(
                f"PP one-step equivalence failed: param={local_param_name}, diff={param_diff_tensor.item():.2e}"
            )
        print("PASS")

    if zero3 is not None:
        zero3.reshard_model()
    dist.destroy_process_group()


def _make_plugins(args: argparse.Namespace):
    case = args.case
    zero3: Zero3Plugin | None = None
    if case == "pp":
        return [PipelineParallelPlugin(schedule=args.pp_schedule)], zero3
    if case == "cp_pp":
        return [ContextParallelPlugin(), PipelineParallelPlugin(schedule=args.pp_schedule)], zero3
    if case == "pp_ddp_sync":
        return [PipelineParallelPlugin(schedule=args.pp_schedule), DataParallelPlugin(async_op=False)], zero3
    if case == "pp_zero1":
        return [PipelineParallelPlugin(schedule=args.pp_schedule), Zero1Plugin(bucket_mb_size=0)], zero3
    if case == "pp_zero2":
        return [PipelineParallelPlugin(schedule=args.pp_schedule), Zero2Plugin(bucket_mb_size=0)], zero3
    if case == "pp_zero3":
        zero3 = Zero3Plugin(wrap_cls={torch.nn.Linear, torch.nn.Embedding, RmsNorm})
        return [PipelineParallelPlugin(schedule=args.pp_schedule), zero3], zero3
    if case == "pp_tp":
        return [TensorParallelPlugin(), PipelineParallelPlugin(schedule=args.pp_schedule)], zero3
    if case == "pp_tp_sp":
        return [TensorParallelPlugin(), SequenceParallelPlugin(), PipelineParallelPlugin(schedule=args.pp_schedule)], zero3
    raise ValueError(f"unknown case={case}")


def _make_baseline_plugins(case: str):
    if case == "pp_tp":
        return [TensorParallelPlugin()]
    if case == "pp_tp_sp":
        return [TensorParallelPlugin(), SequenceParallelPlugin()]
    return None


def _build_runtime_model(case: str):
    if case == "pp_tp":
        return TinyTransformerTp(**_MODEL_KWARGS)
    if case == "pp_tp_sp":
        return TinyTransformerTpSp(**_MODEL_KWARGS)
    return TinyTransformer(**_MODEL_KWARGS)


def _build_baseline_model(case: str):
    if case == "pp_tp":
        return TinyTransformerTp(**_MODEL_KWARGS)
    if case == "pp_tp_sp":
        return TinyTransformerTpSp(**_MODEL_KWARGS)
    return TinyTransformer(**_MODEL_KWARGS)


def _run_baseline(rank: int, args: argparse.Namespace) -> tuple[float, dict[str, torch.Tensor]]:
    baseline_plugins = _make_baseline_plugins(args.case)
    baseline_model, tokens = _build_reference(args.seed, args.batch_size, args.seq_len)
    if baseline_plugins is None:
        optimizer = torch.optim.SGD(baseline_model.parameters(), lr=_LR)
        optimizer.zero_grad(set_to_none=True)
        baseline_loss = baseline_model(causal_lm_batch(tokens))
        baseline_loss.backward()
        optimizer.step()
        return baseline_loss.item(), {name: param.detach() for name, param in baseline_model.named_parameters()}

    sharded_baseline = _build_baseline_model(args.case)
    sharded_baseline.load_state_dict(baseline_model.state_dict())
    baseline_core = RuntimeCore(
        mesh=MeshConfig(dp=args.pp_size, tp=args.tp_size, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=sharded_baseline,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=baseline_plugins,
    )
    baseline_core.setup()
    baseline_loss, should_step = baseline_core.run_step(causal_lm_batch(tokens))
    if not should_step:
        raise AssertionError("baseline TP/SP test expects grad_accum_steps=1, should_step must be True")
    baseline_core.step_optimizer()
    baseline_tp_group = baseline_core.get_group(MeshAxis.TP)
    shard_rules = _rule_by_param_name(baseline_core.model)
    baseline_params = _named_tensors(baseline_core.model, shard_rules, baseline_tp_group)
    return baseline_loss.item(), baseline_params


def main() -> None:
    args = parse_args()
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
