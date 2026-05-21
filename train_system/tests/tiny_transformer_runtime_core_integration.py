"""Integration tests for composing RuntimeCore TP/SP with DP and ZeRO plugins.

The DP cases use a 2x2 mesh: dp=2, tp=2. Ranks in the same DP row share the
same local batch and form a TP group; ranks in the same TP column form a DP
group. This catches the common class of bugs where a plugin accidentally uses
the global world group instead of the intended mesh axis.

Usage:
  PYTHONPATH=. .venv/bin/python train_system/tests/tiny_transformer_runtime_core_integration.py \
    --case tp_sp_ddp_bucket \
    --world-size 4 \
    --dp-size 2 \
    --tp-size 2
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train_system.examples import TinyTransformer, TinyTransformerTpSp
from train_system.parallel import ParallelPlan
from train_system.runtime import MeshAxis, MeshConfig, RuntimeCore
from train_system.runtime.plugins.ddp_v2 import BucketDataParallelPluginV2, DataParallelPluginV2
from train_system.runtime.plugins.sp_v2 import SequenceParallelPluginV2
from train_system.runtime.plugins.tp import ColumnParallelLinear, RowParallelLinear
from train_system.runtime.plugins.tp_v2 import TensorParallelPluginV2
from train_system.runtime.plugins.zero1_v2 import Zero1PluginV2
from train_system.runtime.plugins.zero2_v2 import Zero2PluginV2
from train_system.runtime.plugins.zero3_v2 import Zero3PluginV2


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

_LOSS_ATOL = 1e-3
_STEP_ATOL = 8e-4
_LR = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "tp_sp",
            "tp_sp_ddp_sync",
            "tp_sp_ddp_async",
            "tp_sp_ddp_bucket",
            "tp_sp_zero1",
            "tp_sp_zero2",
            "tp_sp_zero3",
        ),
        required=True,
    )
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29523)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _build_reference(seed: int, batch_size: int, seq_len: int) -> tuple[TinyTransformer, torch.Tensor]:
    torch.manual_seed(seed)
    tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len))
    model = TinyTransformer(**_MODEL_KWARGS)
    return model, tokens


def _mesh_indices(rank: int, tp_size: int) -> tuple[int, int]:
    return rank // tp_size, rank % tp_size


def _rule_by_param_name(model: TinyTransformerTpSp) -> dict[str, str]:
    rules = {}
    for rule in model.parallelize_spec().rules:
        if rule.shard_style in ("col", "row"):
            rules[f"{rule.module_path}.weight"] = rule.shard_style
            rules[f"{rule.module_path}.bias"] = rule.shard_style
    return rules


def _all_gather_tensor(tensor: torch.Tensor, group: dist.ProcessGroup) -> list[torch.Tensor]:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return gathered


def _logical_tensor(
    name: str,
    tensor: torch.Tensor,
    shard_rules: dict[str, str],
    tp_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    shard_style = shard_rules.get(name)
    if shard_style == "col":
        assert tp_group is not None
        return torch.cat(_all_gather_tensor(tensor, tp_group), dim=0)
    if shard_style == "row":
        assert tp_group is not None
        return torch.cat(_all_gather_tensor(tensor, tp_group), dim=1)
    return tensor.detach().clone()


def _logical_named_tensors(
    model: torch.nn.Module,
    shard_rules: dict[str, str],
    tp_group: dist.ProcessGroup | None,
) -> dict[str, torch.Tensor]:
    return {
        name: _logical_tensor(name, param.detach(), shard_rules, tp_group)
        for name, param in model.named_parameters()
    }


def _max_diff(lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]) -> tuple[str, float]:
    worst_name = ""
    worst_diff = 0.0
    for name, lhs_tensor in lhs.items():
        diff = (lhs_tensor - rhs[name]).abs().max().item()
        if diff > worst_diff:
            worst_name = name
            worst_diff = diff
    return worst_name, worst_diff


def _make_plugins(case: str):
    plugins = [TensorParallelPluginV2(), SequenceParallelPluginV2()]
    if case == "tp_sp":
        return plugins, 0
    if case == "tp_sp_ddp_sync":
        return plugins + [DataParallelPluginV2(async_op=False)], 0
    if case == "tp_sp_ddp_async":
        return plugins + [DataParallelPluginV2(async_op=True)], 0
    if case == "tp_sp_ddp_bucket":
        return plugins + [BucketDataParallelPluginV2(bucket_mb_size=0)], 0
    if case == "tp_sp_zero1":
        return plugins + [Zero1PluginV2(bucket_mb_size=0, optimizer_cls=torch.optim.SGD, lr=_LR)], 1
    if case == "tp_sp_zero2":
        return plugins + [Zero2PluginV2(bucket_mb_size=0, optimizer_cls=torch.optim.SGD, lr=_LR)], 2
    if case == "tp_sp_zero3":
        return plugins + [
            Zero3PluginV2(
                wrap_cls={torch.nn.Linear, ColumnParallelLinear, RowParallelLinear},
                optimizer_cls=torch.optim.SGD,
                lr=_LR,
            )
        ], 3
    raise ValueError(f"unknown integration case={case}")


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

    dp_idx, tp_idx = _mesh_indices(rank, args.tp_size)
    baseline_model, full_tokens = _build_reference(args.seed, args.global_batch_size, args.seq_len)
    sharded_model = TinyTransformerTpSp(**_MODEL_KWARGS)
    sharded_model.load_state_dict(baseline_model.state_dict())

    local_batch_size = args.global_batch_size // args.dp_size
    local_tokens = full_tokens.narrow(0, dp_idx * local_batch_size, local_batch_size).contiguous()

    baseline_optimizer = torch.optim.SGD(baseline_model.parameters(), lr=_LR)
    baseline_optimizer.zero_grad(set_to_none=True)
    baseline_train_loss = baseline_model((full_tokens, full_tokens.clone()))
    baseline_train_loss.backward()

    plugins, zero_stage = _make_plugins(args.case)
    core = RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=1, cp=1, ep=1),
        plan=ParallelPlan(zero_stage=zero_stage),
        model=sharded_model,
        optimizer=None,
        plugins=plugins,
    )
    core.setup()
    if zero_stage == 0:
        core.optimizer = torch.optim.SGD(core.model.parameters(), lr=_LR)

    sharded_train_loss = core.run_train_step((local_tokens, local_tokens.clone()))
    baseline_optimizer.step()

    dp_group = core.get_group(MeshAxis.DP)
    avg_loss = sharded_train_loss.detach().clone()
    if dp_group is not None:
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG, group=dp_group)

    zero3_plugin = next((plugin for plugin in core.plugins if isinstance(plugin, Zero3PluginV2)), None)
    if zero3_plugin is not None:
        zero3_plugin.materialize_model()

    shard_rules = _rule_by_param_name(sharded_model)
    tp_group = core.get_group(MeshAxis.TP)
    baseline_params = _logical_named_tensors(baseline_model, {}, None)
    sharded_params = _logical_named_tensors(core.model, shard_rules, tp_group)
    step_name, step_diff = _max_diff(baseline_params, sharded_params)

    if rank == 0:
        loss_diff = abs(baseline_train_loss.item() - avg_loss.item())
        print(f"Case             : {args.case}")
        print(f"Baseline loss    : {baseline_train_loss.item():.6f}")
        print(f"RuntimeCore loss : {avg_loss.item():.6f}")
        print(f"Loss diff        : {loss_diff:.2e}  (atol={_LOSS_ATOL:.2e})")
        print(f"Post-step diff   : {step_diff:.2e}  ({step_name}, atol={_STEP_ATOL:.2e})")
        if loss_diff > _LOSS_ATOL:
            raise AssertionError(f"integration loss mismatch: case={args.case}, diff={loss_diff:.2e}")
        if step_diff > _STEP_ATOL:
            raise AssertionError(
                f"integration one-step mismatch: case={args.case}, param={step_name}, diff={step_diff:.2e}"
            )
        print("PASS")

    if zero3_plugin is not None:
        zero3_plugin.reshard_model()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
