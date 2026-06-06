"""Equivalence test: baseline TinyTransformer vs RuntimeCore CP variants."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from helpers import causal_lm_batch
from models import TinyTransformer, TinyTransformerTp, TinyTransformerTpSp
from parallel.context import ContextParallelAttentionCoreType
from parallel.specs import TpSpShardAxis
from parallel import ParallelPlan
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.plugins.cp import ContextParallelPlugin
from runtime.plugins.ddp import BucketDataParallelPlugin, DataParallelPlugin
from runtime.plugins.sp import SequenceParallelPlugin
from runtime.plugins.tp import TensorParallelPlugin
from runtime.plugins.zero1 import Zero1Plugin
from runtime.plugins.zero2 import Zero2Plugin
from runtime.plugins.zero3 import Zero3Plugin
from models.tiny_transformer import RmsNorm


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
_GRAD_ATOL = 1e-2
_STEP_ATOL = 1e-4
_LR = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("cp", "cp_ddp_sync", "cp_ddp_async", "cp_ddp_bucket", "cp_zero1", "cp_zero2", "cp_zero3"),
        default="cp",
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--use-sp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--cp-attn-core",
        type=str,
        default="all_gather_kv",
        choices=tuple(core.value for core in ContextParallelAttentionCoreType),
    )
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29581)
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


def _build_sharded_model(args: argparse.Namespace) -> TinyTransformer:
    if args.tp_size > 1 and args.use_sp:
        return TinyTransformerTpSp(**_MODEL_KWARGS)
    if args.tp_size > 1:
        return TinyTransformerTp(**_MODEL_KWARGS)
    return TinyTransformer(**_MODEL_KWARGS)


def _logical_named_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def _rule_by_param_name(model: TinyTransformer) -> dict[str, str]:
    if not hasattr(model, "tpsp_parallelize_spec"):
        return {}
    rules = {}
    for rule in model.tpsp_parallelize_spec().rules:
        if rule.shard_axis in (TpSpShardAxis.PARAM_OUT, TpSpShardAxis.PARAM_IN):
            rules[f"{rule.module_path}.weight"] = rule.shard_axis
            rules[f"{rule.module_path}.bias"] = rule.shard_axis
    return rules


def _all_gather_tensor(tensor: torch.Tensor, group: dist.ProcessGroup | None) -> list[torch.Tensor]:
    if group is None:
        return [tensor.detach().clone()]
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
        return torch.cat(_all_gather_tensor(tensor, tp_group), dim=0)
    if shard_axis == TpSpShardAxis.PARAM_IN:
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


def _logical_named_sharded_grads(
    model: torch.nn.Module,
    shard_rules: dict[str, str],
    tp_group: dist.ProcessGroup | None,
) -> dict[str, torch.Tensor]:
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grads[name] = _logical_tensor(name, param.grad.detach(), shard_rules, tp_group)
    return grads


def _logical_named_zero_shard_grads(core: RuntimeCore, zero_plugin: Zero1Plugin | Zero2Plugin | Zero3Plugin) -> dict[str, torch.Tensor]:
    dp_group = core.get_group(MeshAxis.DP)
    if dp_group is None:
        raise ValueError("ZeRO1 grad materialization requires DP group")
    name_by_param = {id(param): name for name, param in core.model.named_parameters()}
    logical_grads: dict[str, torch.Tensor] = {}
    for bucket in zero_plugin.buckets:
        local_grad = bucket.local_param.grad
        if local_grad is None:
            continue
        gathered = [torch.empty_like(local_grad) for _ in range(dist.get_world_size(dp_group))]
        dist.all_gather(gathered, local_grad.contiguous(), group=dp_group)
        full_grad = torch.cat(gathered, dim=0)
        offset = 0
        logical_names = getattr(bucket, "logical_names", None)
        param_shapes = getattr(bucket, "param_shapes", None)
        param_numels = getattr(bucket, "param_numels", None)
        if logical_names is not None and param_shapes is not None and param_numels is not None:
            for name, shape, numel in zip(logical_names, param_shapes, param_numels):
                logical_grads[name] = full_grad[offset : offset + numel].view(shape).detach().clone()
                offset += numel
            continue
        for param in bucket.params:
            name = name_by_param[id(param)]
            numel = param.numel()
            logical_grads[name] = full_grad[offset : offset + numel].view_as(param).detach().clone()
            offset += numel
    return logical_grads


def _logical_named_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
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


def _make_baseline_core(reference_model: TinyTransformer, args: argparse.Namespace) -> RuntimeCore:
    model = TinyTransformer(**_MODEL_KWARGS)
    model.load_state_dict(reference_model.state_dict())
    plugins = [DataParallelPlugin(async_op=False)] if args.case in {
        "cp_ddp_sync",
        "cp_ddp_async",
        "cp_ddp_bucket",
        "cp_zero1",
        "cp_zero2",
        "cp_zero3",
    } else []
    if args.tp_size > 1:
        plugins.append(TensorParallelPlugin())
    if args.use_sp:
        plugins.append(SequenceParallelPlugin())
    return RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, pp=1, cp=args.cp_size, tp=args.tp_size, ep=1),
        plan=ParallelPlan(),
        model=model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=plugins,
    )


def _make_runtime_core(reference_model: TinyTransformer, args: argparse.Namespace) -> RuntimeCore:
    model = TinyTransformer(**_MODEL_KWARGS)
    model.load_state_dict(reference_model.state_dict())
    plugins = [ContextParallelPlugin()]
    if args.tp_size > 1:
        plugins.insert(0, TensorParallelPlugin())
    if args.use_sp:
        plugins.insert(1 if args.tp_size > 1 else 0, SequenceParallelPlugin())
    if args.case == "cp_ddp_sync":
        plugins.append(DataParallelPlugin(async_op=False))
    if args.case == "cp_ddp_async":
        plugins.append(DataParallelPlugin(async_op=True))
    if args.case == "cp_ddp_bucket":
        plugins.append(BucketDataParallelPlugin(bucket_mb_size=0))
    if args.case == "cp_zero1":
        plugins.append(Zero1Plugin(bucket_mb_size=0))
    if args.case == "cp_zero2":
        plugins.append(Zero2Plugin(bucket_mb_size=0))
    if args.case == "cp_zero3":
        plugins.append(Zero3Plugin(wrap_cls={torch.nn.Linear, torch.nn.Embedding, RmsNorm}))
    return RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, pp=1, cp=args.cp_size, tp=args.tp_size, ep=1),
        plan=ParallelPlan(
            zero_stage=1 if args.case == "cp_zero1" else 2 if args.case == "cp_zero2" else 3 if args.case == "cp_zero3" else 0,
            cp_attn_core=ContextParallelAttentionCoreType(args.cp_attn_core),
        ),
        model=model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=plugins,
    )


def _reduce_runtime_loss(loss: torch.Tensor, core: RuntimeCore) -> float:
    reduced = loss.detach().clone()
    cp_group = core.get_group(MeshAxis.CP)
    dp_group = core.get_group(MeshAxis.DP)
    if cp_group is not None:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=cp_group)
    if dp_group is not None:
        dist.all_reduce(reduced, op=dist.ReduceOp.AVG, group=dp_group)
    return float(reduced.item())


def _reduce_baseline_loss(loss: torch.Tensor, core: RuntimeCore) -> float:
    reduced = loss.detach().clone()
    dp_group = core.get_group(MeshAxis.DP)
    if dp_group is not None:
        dist.all_reduce(reduced, op=dist.ReduceOp.AVG, group=dp_group)
    return float(reduced.item())


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )
    if args.world_size != args.dp_size * args.cp_size * args.tp_size:
        raise ValueError("CP equivalence expects world_size == dp_size * cp_size * tp_size")
    if args.seq_len % args.cp_size != 0:
        raise ValueError("seq_len must be divisible by cp_size")
    if args.batch_size % args.dp_size != 0:
        raise ValueError("batch_size must be divisible by dp_size")

    reference_model, tokens = _build_reference(args.seed, args.batch_size, args.seq_len)
    baseline_core = _make_baseline_core(reference_model, args)
    runtime_core = _make_runtime_core(reference_model, args)
    baseline_core.setup()
    runtime_core.setup()

    dp_idx = rank // (args.cp_size * args.tp_size)
    local_batch_size = args.batch_size // args.dp_size
    local_tokens = tokens.narrow(0, dp_idx * local_batch_size, local_batch_size).contiguous()
    batch = causal_lm_batch(local_tokens)

    baseline_loss, baseline_should_step = baseline_core.run_step(batch)
    runtime_loss, runtime_should_step = runtime_core.run_step(batch)
    if not baseline_should_step or not runtime_should_step:
        raise AssertionError("CP equivalence expects grad_accum_steps=1")

    baseline_loss_value = _reduce_baseline_loss(baseline_loss, baseline_core)
    runtime_loss_value = _reduce_runtime_loss(runtime_loss, runtime_core)

    baseline_tp_group = baseline_core.get_group(MeshAxis.TP)
    runtime_tp_group = runtime_core.get_group(MeshAxis.TP)
    baseline_shard_rules = _rule_by_param_name(baseline_core.model)
    runtime_shard_rules = _rule_by_param_name(runtime_core.model)
    baseline_grads = _logical_named_sharded_grads(baseline_core.model, baseline_shard_rules, baseline_tp_group)
    zero_plugin = next(
        (plugin for plugin in runtime_core.plugins if isinstance(plugin, (Zero1Plugin, Zero2Plugin, Zero3Plugin))),
        None,
    )
    runtime_grads = (
        _logical_named_zero_shard_grads(runtime_core, zero_plugin)
        if zero_plugin is not None
        else _logical_named_sharded_grads(runtime_core.model, runtime_shard_rules, runtime_tp_group)
    )
    grad_name, grad_diff = _max_diff(runtime_grads, baseline_grads)

    baseline_core.step_optimizer()
    runtime_core.step_optimizer()
    zero3 = next((plugin for plugin in runtime_core.plugins if isinstance(plugin, Zero3Plugin)), None)
    if zero3 is not None:
        zero3.materialize_model()
    baseline_params = _logical_named_tensors(baseline_core.model, baseline_shard_rules, baseline_tp_group)
    runtime_params = _logical_named_tensors(runtime_core.model, runtime_shard_rules, runtime_tp_group)
    step_name, step_diff = _max_diff(runtime_params, baseline_params)

    loss_diff_tensor = torch.tensor(abs(runtime_loss_value - baseline_loss_value), dtype=torch.float64)
    grad_diff_tensor = torch.tensor(grad_diff, dtype=torch.float64)
    step_diff_tensor = torch.tensor(step_diff, dtype=torch.float64)
    dist.all_reduce(loss_diff_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(grad_diff_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(step_diff_tensor, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"Case          : {args.case}")
        print(f"CP attn core  : {args.cp_attn_core}")
        print(f"Baseline loss : {baseline_loss_value:.6f}")
        print(f"Runtime CP    : {runtime_loss_value:.6f}")
        print(f"Loss diff     : {loss_diff_tensor.item():.2e}  (atol={_LOSS_ATOL:.2e})")
        print(f"Grad diff     : {grad_diff_tensor.item():.2e}  ({grad_name}, atol={_GRAD_ATOL:.2e})")
        print(f"Step diff     : {step_diff_tensor.item():.2e}  ({step_name}, atol={_STEP_ATOL:.2e})")
        if loss_diff_tensor.item() > _LOSS_ATOL:
            raise AssertionError(f"CP loss equivalence failed: diff={loss_diff_tensor.item():.2e}")
        if grad_diff_tensor.item() > _GRAD_ATOL:
            raise AssertionError(f"CP gradient equivalence failed: param={grad_name}, diff={grad_diff_tensor.item():.2e}")
        if step_diff_tensor.item() > _STEP_ATOL:
            raise AssertionError(f"CP one-step equivalence failed: param={step_name}, diff={step_diff_tensor.item():.2e}")
        print("PASS")

    if zero3 is not None:
        zero3.reshard_model()

    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
