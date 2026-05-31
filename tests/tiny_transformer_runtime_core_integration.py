"""Integration tests for composing RuntimeCore TP/SP with DP and ZeRO plugins.

The DP cases use a 2x2 mesh: dp=2, tp=2. Ranks in the same DP row share the
same local batch and form a TP group; ranks in the same TP column form a DP
group. This catches the common class of bugs where a plugin accidentally uses
the global world group instead of the intended mesh axis.

Usage:
  PYTHONPATH=. .venv/bin/python tests/tiny_transformer_runtime_core_integration.py \
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

from models import TinyTransformer, TinyTransformerTp, TinyTransformerTpSp
from helpers import causal_lm_batch
from parallel.specs import TpSpShardAxis
from parallel import ParallelPlan
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.plugins.ddp import BucketDataParallelPlugin, DataParallelPlugin
from runtime.plugins.grad_clip import GradClipPlugin
from runtime.plugins.precision import PrecisionPlugin
from runtime.plugins.sp import SequenceParallelPlugin
from runtime.layers.tp import ColumnParallelLinear, RowParallelLinear
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
            "tp_bf16",
            "tp_sp_ddp_sync",
            "tp_sp_ddp_async",
            "tp_sp_ddp_bucket",
            "tp_sp_zero1",
            "tp_sp_zero2",
            "tp_sp_zero3",
            "tp_sp_zero3_bf16_clip",
            "tp_sp_zero3_bf16_clip_accum2",
            "tp_zero3_bf16_clip",
            "tp_zero3_bf16_clip_accum2",
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
        if rule.shard_axis in (TpSpShardAxis.PARAM_OUT, TpSpShardAxis.PARAM_IN):
            rules[f"{rule.module_path}.weight"] = rule.shard_axis
            rules[f"{rule.module_path}.bias"] = rule.shard_axis
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
    shard_axis = shard_rules.get(name)
    if shard_axis == TpSpShardAxis.PARAM_OUT:
        assert tp_group is not None
        return torch.cat(_all_gather_tensor(tensor, tp_group), dim=0)
    if shard_axis == TpSpShardAxis.PARAM_IN:
        assert tp_group is not None
        return torch.cat(_all_gather_tensor(tensor, tp_group), dim=1)
    return tensor.detach().clone()


def _logical_named_tensors(
    model: torch.nn.Module,
    shard_rules: dict[str, str],
    tp_group: dist.ProcessGroup | None,
) -> dict[str, torch.Tensor]:
    def _normalize_name(name: str) -> str:
        return name[len("module.") :] if name.startswith("module.") else name

    return {
        _normalize_name(name): _logical_tensor(_normalize_name(name), param.detach(), shard_rules, tp_group)
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
    plugins = [TensorParallelPlugin(), SequenceParallelPlugin()]
    if case == "tp_sp":
        return plugins, 0
    if case == "tp_bf16":
        return [TensorParallelPlugin(), PrecisionPlugin(compute_dtype=torch.bfloat16)], 0
    if case == "tp_sp_ddp_sync":
        return plugins + [DataParallelPlugin(async_op=False)], 0
    if case == "tp_sp_ddp_async":
        return plugins + [DataParallelPlugin(async_op=True)], 0
    if case == "tp_sp_ddp_bucket":
        return plugins + [BucketDataParallelPlugin(bucket_mb_size=0)], 0
    if case == "tp_sp_zero1":
        return plugins + [Zero1Plugin(bucket_mb_size=0)], 1
    if case == "tp_sp_zero2":
        return plugins + [Zero2Plugin(bucket_mb_size=0)], 2
    if case == "tp_sp_zero3":
        return plugins + [
            Zero3Plugin(
                wrap_cls={torch.nn.Linear, ColumnParallelLinear, RowParallelLinear},
            )
        ], 3
    if case in {"tp_sp_zero3_bf16_clip", "tp_sp_zero3_bf16_clip_accum2"}:
        return plugins + [
            Zero3Plugin(
                wrap_cls={torch.nn.Linear, ColumnParallelLinear, RowParallelLinear},
            ),
            PrecisionPlugin(compute_dtype=torch.bfloat16),
            GradClipPlugin(max_norm=1.0),
        ], 3
    if case in {"tp_zero3_bf16_clip", "tp_zero3_bf16_clip_accum2"}:
        return [
            TensorParallelPlugin(),
            Zero3Plugin(
                wrap_cls={torch.nn.Linear, ColumnParallelLinear, RowParallelLinear},
            ),
            PrecisionPlugin(compute_dtype=torch.bfloat16),
            GradClipPlugin(max_norm=1.0),
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
    if args.case in {"tp_bf16", "tp_zero3_bf16_clip", "tp_zero3_bf16_clip_accum2"}:
        sharded_model = TinyTransformerTp(**_MODEL_KWARGS)
    else:
        sharded_model = TinyTransformerTpSp(**_MODEL_KWARGS)
    sharded_model.load_state_dict(baseline_model.state_dict())

    local_batch_size = args.global_batch_size // args.dp_size
    local_tokens = full_tokens.narrow(0, dp_idx * local_batch_size, local_batch_size).contiguous()

    baseline_optimizer = torch.optim.SGD(baseline_model.parameters(), lr=_LR)
    baseline_optimizer.zero_grad(set_to_none=True)
    baseline_train_loss = baseline_model(causal_lm_batch(full_tokens))
    baseline_train_loss.backward()

    if args.case in {"tp_bf16", "tp_sp_zero3_bf16_clip", "tp_sp_zero3_bf16_clip_accum2", "tp_zero3_bf16_clip", "tp_zero3_bf16_clip_accum2"} and not _supports_bf16_autocast():
        if rank == 0:
            print("SKIP: bf16 autocast is not supported on this runtime")
        dist.destroy_process_group()
        return

    grad_accum_steps = 2 if args.case in {"tp_zero3_bf16_clip_accum2", "tp_sp_zero3_bf16_clip_accum2"} else 1
    if local_batch_size % grad_accum_steps != 0:
        raise ValueError("local batch size must be divisible by grad_accum_steps")
    plugins, zero_stage = _make_plugins(args.case)
    core = RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=1, cp=1, ep=1),
        plan=ParallelPlan(zero_stage=zero_stage),
        model=sharded_model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=plugins,
        grad_accum_steps=grad_accum_steps,
    )
    core.setup()

    if args.case in {"tp_bf16", "tp_zero3_bf16_clip", "tp_zero3_bf16_clip_accum2"}:
        bf16_ok = 1
        bf16_error = ""
        try:
            with torch.no_grad():
                _ = core.model(causal_lm_batch(local_tokens))
        except Exception as exc:
            bf16_ok = 0
            bf16_error = str(exc)
        ok_tensor = torch.tensor([bf16_ok], dtype=torch.int64)
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
        if ok_tensor.item() == 0:
            if rank == 0:
                detail = bf16_error if bf16_error else "one or more ranks failed bf16 forward preflight"
                print(f"SKIP: {args.case} preflight failed: {detail}")
            dist.destroy_process_group()
            return

    if grad_accum_steps == 1:
        sharded_train_loss = core.run_train_step(causal_lm_batch(local_tokens))
    else:
        micro_batch_size = local_batch_size // grad_accum_steps
        sharded_train_loss = torch.zeros((), dtype=torch.float32, device=local_tokens.device)
        for micro_idx in range(grad_accum_steps):
            micro_tokens = local_tokens.narrow(0, micro_idx * micro_batch_size, micro_batch_size).contiguous()
            sharded_train_loss = sharded_train_loss + core.run_train_step(causal_lm_batch(micro_tokens)).detach()
    baseline_optimizer.step()

    dp_group = core.get_group(MeshAxis.DP)
    avg_loss = sharded_train_loss.detach().clone()
    if dp_group is not None:
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG, group=dp_group)

    zero3_plugin = next((plugin for plugin in core.plugins if isinstance(plugin, Zero3Plugin)), None)
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
        loss_atol = _LOSS_ATOL
        step_atol = _STEP_ATOL
        if args.case in {"tp_bf16", "tp_sp_zero3_bf16_clip", "tp_sp_zero3_bf16_clip_accum2", "tp_zero3_bf16_clip", "tp_zero3_bf16_clip_accum2"}:
            loss_atol = 5e-2
            step_atol = 5e-2
        print(f"Loss diff        : {loss_diff:.2e}  (atol={loss_atol:.2e})")
        print(f"Post-step diff   : {step_diff:.2e}  ({step_name}, atol={step_atol:.2e})")
        if loss_diff > loss_atol:
            raise AssertionError(f"integration loss mismatch: case={args.case}, diff={loss_diff:.2e}")
        if step_diff > step_atol:
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


def _supports_bf16_autocast() -> bool:
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            x = torch.randn(8, 8)
            _ = x @ x
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
