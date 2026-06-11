"""Equivalence test: baseline TinyMoETransformer vs RuntimeCore EP variants."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distributed_test_utils import (
    max_diff as _max_diff,
    moe_named_tensors as _moe_named_tensors,
    moe_zero_named_grads as _moe_zero_named_grads,
    named_tensors as _named_tensors,
    reduce_loss as _reduce_loss,
    rule_by_param_name as _rule_by_param_name,
)
from helpers import causal_lm_batch
from models import TinyMoETransformer, TinyMoETransformerTp, TinyMoETransformerTpSp
from models.tiny_transformer import RmsNorm
from parallel import ParallelPlan
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.core import RuntimePhase
from runtime.plugins.ddp import BucketDataParallelPlugin, DataParallelPlugin
from runtime.plugins.ep import ExpertParallelPlugin
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
    n_layers=2,
    vocab_size=256,
    max_seq_len=64,
    num_experts=4,
)

_LOSS_ATOL = 1e-3
_GRAD_ATOL = 2e-2
_STEP_ATOL = 5e-4
_LR = 1e-2


def _loss_atol(case: str) -> float:
    return 5e-3 if case in {"ep_tp", "ep_tp_sp"} else _LOSS_ATOL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "ep",
            "ep_tp",
            "ep_tp_sp",
            "ep_ddp_sync",
            "ep_ddp_async",
            "ep_ddp_bucket",
            "ep_zero1",
            "ep_zero2",
            "ep_zero3",
            "ep_tp_sp_zero1",
            "ep_tp_sp_zero2",
            "ep_tp_sp_zero3",
        ),
        default="ep",
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29590)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _tokens_for_dp(seed: int, dp_idx: int, batch_size: int, seq_len: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed + dp_idx)
    return torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len), generator=generator)


def _build_reference(seed: int, dp_size: int, batch_size: int, seq_len: int) -> tuple[TinyMoETransformer, torch.Tensor]:
    torch.manual_seed(seed)
    model = TinyMoETransformer(**_MODEL_KWARGS)
    tokens = torch.cat([_tokens_for_dp(seed, dp_idx, batch_size, seq_len) for dp_idx in range(dp_size)], dim=0)
    return model, tokens

def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )

    mesh = MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=1, cp=1, ep=args.ep_size)
    dp_idx, _, _, _ = mesh.rank_coordinates(rank)
    local_tokens = _tokens_for_dp(args.seed, dp_idx, args.batch_size, args.seq_len)

    baseline_model, global_tokens = _build_reference(args.seed, args.dp_size, args.batch_size, args.seq_len)
    baseline_model.train()
    baseline_optimizer = torch.optim.SGD(baseline_model.parameters(), lr=_LR)
    baseline_optimizer.zero_grad(set_to_none=True)
    baseline_loss = baseline_model(causal_lm_batch(global_tokens))
    baseline_loss.backward()

    if args.case in {
        "ep",
        "ep_ddp_sync",
        "ep_ddp_async",
        "ep_ddp_bucket",
        "ep_zero1",
        "ep_zero2",
        "ep_zero3",
        "ep_tp_sp_zero1",
        "ep_tp_sp_zero2",
        "ep_tp_sp_zero3",
    }:
        sharded_cls = TinyMoETransformer
    elif args.case == "ep_tp":
        sharded_cls = TinyMoETransformerTp
    else:
        sharded_cls = TinyMoETransformerTpSp
    sharded_model = sharded_cls(**_MODEL_KWARGS)
    sharded_model.load_state_dict(baseline_model.state_dict())
    plugins = [ExpertParallelPlugin()]
    zero_plugin: Zero1Plugin | Zero2Plugin | Zero3Plugin | None = None
    if args.case == "ep_ddp_sync":
        plugins.append(DataParallelPlugin(async_op=False))
    elif args.case == "ep_ddp_async":
        plugins.append(DataParallelPlugin(async_op=True))
    elif args.case == "ep_ddp_bucket":
        plugins.append(BucketDataParallelPlugin(bucket_mb_size=0))
    elif args.case == "ep_zero1":
        zero_plugin = Zero1Plugin(bucket_mb_size=0)
        plugins.append(zero_plugin)
    elif args.case == "ep_zero2":
        zero_plugin = Zero2Plugin(bucket_mb_size=0)
        plugins.append(zero_plugin)
    elif args.case == "ep_zero3":
        zero_plugin = Zero3Plugin(wrap_cls={torch.nn.Linear, torch.nn.Embedding, RmsNorm})
        plugins.append(zero_plugin)
    elif args.case == "ep_tp_sp_zero1":
        zero_plugin = Zero1Plugin(bucket_mb_size=0)
        plugins.append(zero_plugin)
    elif args.case == "ep_tp_sp_zero2":
        zero_plugin = Zero2Plugin(bucket_mb_size=0)
        plugins.append(zero_plugin)
    elif args.case == "ep_tp_sp_zero3":
        zero_plugin = Zero3Plugin(wrap_cls={torch.nn.Linear, torch.nn.Embedding, RmsNorm})
        plugins.append(zero_plugin)
    if args.case in {"ep_tp", "ep_tp_sp", "ep_tp_sp_zero1", "ep_tp_sp_zero2", "ep_tp_sp_zero3"}:
        plugins.insert(0, TensorParallelPlugin())
    if args.case in {"ep_tp_sp", "ep_tp_sp_zero1", "ep_tp_sp_zero2", "ep_tp_sp_zero3"}:
        plugins.insert(1, SequenceParallelPlugin())
    core = RuntimeCore(
        mesh=mesh,
        plan=ParallelPlan(),
        model=sharded_model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=plugins,
    )
    core.setup()
    core.model.train()

    loss, should_step = core.run_step(causal_lm_batch(local_tokens))
    if not should_step:
        raise AssertionError("EP equivalence test expected should_step=True")
    runtime_loss = _reduce_loss(loss, core)

    # Some plugin paths intentionally launch async grad sync in POST_BACKWARD and
    # only guarantee completion by PRE_STEP. Flush that boundary before comparing
    # gradients for equivalence.
    core._run_phase(RuntimePhase.PRE_STEP)

    shard_rules = _rule_by_param_name(sharded_model)
    tp_group = core.get_group(MeshAxis.TP)
    ep_group = core.get_group(MeshAxis.EP)
    baseline_grads = _named_tensors(baseline_model, {}, None, grads=True)
    runtime_grads = (
        _moe_zero_named_grads(core.model, zero_plugin, shard_rules, tp_group, ep_group)
        if zero_plugin is not None
        else _moe_named_tensors(core.model, shard_rules, tp_group, ep_group, grads=True)
    )
    grad_name, grad_diff = _max_diff(baseline_grads, runtime_grads)

    baseline_optimizer.step()
    core.step_optimizer()

    if zero_plugin is not None and isinstance(zero_plugin, Zero3Plugin):
        zero_plugin.materialize_model()
    baseline_params = _named_tensors(baseline_model, {}, None)
    runtime_params = _moe_named_tensors(core.model, shard_rules, tp_group, ep_group)
    step_name, step_diff = _max_diff(baseline_params, runtime_params)
    if zero_plugin is not None and isinstance(zero_plugin, Zero3Plugin):
        zero_plugin.reshard_model()

    if rank == 0:
        loss_atol = _loss_atol(args.case)
        diff = abs(baseline_loss.item() - runtime_loss.item())
        if args.case == "ep_tp":
            runtime_label = "RuntimeCore EP+TP"
        elif args.case == "ep_tp_sp":
            runtime_label = "RuntimeCore EP+TP+SP"
        elif args.case == "ep_ddp_sync":
            runtime_label = "RuntimeCore EP+DDP(sync)"
        elif args.case == "ep_ddp_async":
            runtime_label = "RuntimeCore EP+DDP(async)"
        elif args.case == "ep_ddp_bucket":
            runtime_label = "RuntimeCore EP+DDP(bucket)"
        elif args.case == "ep_zero1":
            runtime_label = "RuntimeCore EP+ZeRO1"
        elif args.case == "ep_zero2":
            runtime_label = "RuntimeCore EP+ZeRO2"
        elif args.case == "ep_zero3":
            runtime_label = "RuntimeCore EP+ZeRO3"
        elif args.case == "ep_tp_sp_zero1":
            runtime_label = "RuntimeCore EP+TP+SP+ZeRO1"
        elif args.case == "ep_tp_sp_zero2":
            runtime_label = "RuntimeCore EP+TP+SP+ZeRO2"
        elif args.case == "ep_tp_sp_zero3":
            runtime_label = "RuntimeCore EP+TP+SP+ZeRO3"
        else:
            runtime_label = "RuntimeCore EP"
        print(f"Baseline loss : {baseline_loss.item():.6f}")
        print(f"{runtime_label}: {runtime_loss.item():.6f}")
        print(f"Diff          : {diff:.2e}  (atol={loss_atol:.2e})")
        print(f"Grad diff     : {grad_diff:.2e}  ({grad_name}, atol={_GRAD_ATOL:.2e})")
        print(f"Step diff     : {step_diff:.2e}  ({step_name}, atol={_STEP_ATOL:.2e})")
        if diff > loss_atol:
            raise AssertionError(
                f"RuntimeCore EP equivalence failed: baseline_loss={baseline_loss.item():.6f}, "
                f"runtime_loss={runtime_loss.item():.6f}, diff={diff:.2e}, atol={loss_atol:.2e}"
            )
        if grad_diff > _GRAD_ATOL:
            raise AssertionError(
                f"RuntimeCore EP gradient equivalence failed: param={grad_name}, "
                f"diff={grad_diff:.2e}, atol={_GRAD_ATOL:.2e}"
            )
        if step_diff > _STEP_ATOL:
            raise AssertionError(
                f"RuntimeCore EP one-step equivalence failed: param={step_name}, "
                f"diff={step_diff:.2e}, atol={_STEP_ATOL:.2e}"
            )
        print("PASS")

    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    assert args.world_size == args.dp_size * args.tp_size
    if args.case in {"ep", "ep_ddp_sync", "ep_ddp_async", "ep_ddp_bucket", "ep_zero1", "ep_zero2", "ep_zero3"}:
        assert args.tp_size == 1
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
