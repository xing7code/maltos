"""Equivalence test: baseline TinyMoETransformer vs RuntimeCore EP variants."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from helpers import causal_lm_batch
from models import TinyMoETransformer, TinyMoETransformerTp, TinyMoETransformerTpSp
from models.tiny_transformer import RmsNorm
from parallel import ParallelPlan
from parallel.specs import TpSpShardAxis
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.core import RuntimePhase
from runtime.plugins.ddp import BucketDataParallelPlugin, DataParallelPlugin
from runtime.plugins.ep import ExpertParallelPlugin, _ExpertParallelMoE
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


def _rule_by_param_name(model: TinyMoETransformer) -> dict[str, str]:
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


def _runtime_local_expert_tensors(model: torch.nn.Module, *, grads: bool) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, _ExpertParallelMoE):
            continue
        for local_idx, global_idx in enumerate(module.local_expert_ids):
            expert = module.local_experts[local_idx]
            for param_name, param in expert.named_parameters():
                full_name = f"{module_name}.experts.{global_idx}.{param_name}"
                if grads:
                    tensor = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param)
                else:
                    tensor = param.detach().clone()
                tensors[full_name] = tensor.cpu()
    return tensors


def _bucket_grad_by_param_id(
    zero_plugin: Zero1Plugin | Zero2Plugin | Zero3Plugin,
) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for bucket in zero_plugin.buckets:
        if bucket.local_param.grad is None:
            continue
        shard_group = bucket.group_context.group
        if shard_group is not None:
            full_grad = torch.cat(_all_gather_tensor(bucket.local_param.grad.detach(), shard_group), dim=0)
        else:
            full_grad = bucket.local_param.grad.detach().clone()
        if hasattr(bucket, "param_numels"):
            offset = 0
            for param, numel, shape in zip(bucket.params, bucket.param_numels, bucket.param_shapes):
                result[id(param)] = full_grad[offset : offset + numel].view(shape).cpu()
                offset += numel
            continue
        offset = 0
        for param in bucket.params:
            numel = param.numel()
            result[id(param)] = full_grad[offset : offset + numel].view(param.shape).cpu()
            offset += numel
    return result


def _runtime_local_expert_zero_grads(
    model: torch.nn.Module,
    grad_by_pid: dict[int, torch.Tensor],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, _ExpertParallelMoE):
            continue
        for local_idx, global_idx in enumerate(module.local_expert_ids):
            expert = module.local_experts[local_idx]
            for param_name, param in expert.named_parameters():
                full_name = f"{module_name}.experts.{global_idx}.{param_name}"
                tensors[full_name] = grad_by_pid.get(id(param), torch.zeros_like(param).cpu())
    return tensors


def _runtime_logical_named_tensors(
    model: torch.nn.Module,
    shard_rules: dict[str, str],
    tp_group: dist.ProcessGroup | None,
    ep_group: dist.ProcessGroup | None,
    *,
    grads: bool,
) -> dict[str, torch.Tensor]:
    tensors = {}
    for name, param in model.named_parameters():
        if ".local_experts." in name:
            continue
        source = param.grad if grads else param.detach()
        if source is None:
            source = torch.zeros_like(param)
        tensors[name] = _logical_tensor(name, source.detach(), shard_rules, tp_group)
    if ep_group is not None:
        gathered: list[dict[str, torch.Tensor]] = [None for _ in range(dist.get_world_size(ep_group))]  # type: ignore[list-item]
        dist.all_gather_object(gathered, _runtime_local_expert_tensors(model, grads=grads), group=ep_group)
        for shard in gathered:
            for name, tensor in shard.items():
                tensors[name] = tensor.to(next(model.parameters()).device)
    return tensors


def _runtime_logical_named_zero_grads(
    model: torch.nn.Module,
    zero_plugin: Zero1Plugin | Zero2Plugin | Zero3Plugin,
    shard_rules: dict[str, str],
    tp_group: dist.ProcessGroup | None,
    ep_group: dist.ProcessGroup | None,
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    grad_by_pid = _bucket_grad_by_param_id(zero_plugin)
    tensors: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if ".local_experts." in name:
            continue
        grad = grad_by_pid.get(id(param), torch.zeros_like(param).cpu())
        tensors[name] = _logical_tensor(name, grad.to(device), shard_rules, tp_group)
    if ep_group is not None:
        gathered: list[dict[str, torch.Tensor]] = [None for _ in range(dist.get_world_size(ep_group))]  # type: ignore[list-item]
        dist.all_gather_object(gathered, _runtime_local_expert_zero_grads(model, grad_by_pid), group=ep_group)
        for shard in gathered:
            for name, tensor in shard.items():
                tensors[name] = tensor.to(device)
    return tensors


def _baseline_named_tensors(model: torch.nn.Module, *, grads: bool) -> dict[str, torch.Tensor]:
    tensors = {}
    for name, param in model.named_parameters():
        source = param.grad if grads else param.detach()
        if source is None:
            source = torch.zeros_like(param)
        tensors[name] = source.detach().clone()
    return tensors


def _max_diff(lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]) -> tuple[str, float]:
    worst_name = ""
    worst_diff = 0.0
    for name, lhs_tensor in lhs.items():
        rhs_tensor = rhs[name].to(lhs_tensor.device, lhs_tensor.dtype)
        diff = (lhs_tensor - rhs_tensor).abs().max().item()
        if diff > worst_diff:
            worst_name = name
            worst_diff = diff
    return worst_name, worst_diff


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
    runtime_loss = loss.detach().clone()
    dp_group = core.get_group(MeshAxis.DP)
    if dp_group is not None:
        dist.all_reduce(runtime_loss, op=dist.ReduceOp.AVG, group=dp_group)

    # Some plugin paths intentionally launch async grad sync in POST_BACKWARD and
    # only guarantee completion by PRE_STEP. Flush that boundary before comparing
    # gradients for equivalence.
    core._run_phase(RuntimePhase.PRE_STEP)

    shard_rules = _rule_by_param_name(sharded_model)
    tp_group = core.get_group(MeshAxis.TP)
    ep_group = core.get_group(MeshAxis.EP)
    baseline_grads = _baseline_named_tensors(baseline_model, grads=True)
    runtime_grads = (
        _runtime_logical_named_zero_grads(core.model, zero_plugin, shard_rules, tp_group, ep_group)
        if zero_plugin is not None
        else _runtime_logical_named_tensors(core.model, shard_rules, tp_group, ep_group, grads=True)
    )
    grad_name, grad_diff = _max_diff(baseline_grads, runtime_grads)

    baseline_optimizer.step()
    core.step_optimizer()

    if zero_plugin is not None and isinstance(zero_plugin, Zero3Plugin):
        zero_plugin.materialize_model()
    baseline_params = _baseline_named_tensors(baseline_model, grads=False)
    runtime_params = _runtime_logical_named_tensors(core.model, shard_rules, tp_group, ep_group, grads=False)
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
