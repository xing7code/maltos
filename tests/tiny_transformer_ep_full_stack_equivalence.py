"""Equivalence test for the EP full stack: PP+CP+TP+SP+EP+ZeRO3."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from helpers import causal_lm_batch
from models import TinyMoETransformer, TinyMoETransformerTpSp
from models.tiny_transformer import RmsNorm
from parallel import ParallelPlan
from parallel.schedule import PipelineScheduleConfig
from parallel.specs import TpSpShardAxis
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.core import RuntimePhase
from runtime.plugins.cp import ContextParallelPlugin
from runtime.plugins.ep import ExpertParallelPlugin, _ExpertParallelMoE
from runtime.plugins.pp import PipelineParallelPlugin
from runtime.plugins.sp import SequenceParallelPlugin
from runtime.plugins.tp import TensorParallelPlugin
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
    num_experts=4,
)

_ZERO3_WRAP_CLS = {torch.nn.Linear, torch.nn.Embedding, RmsNorm}
_LOSS_ATOL = 5e-3
_GRAD_ATOL = 2e-2
_STEP_ATOL = 5e-4
_LR = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--cp-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--pp-microbatches", type=int, default=2)
    parser.add_argument("--pp-schedule", choices=("afab", "1f1b"), default="afab")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29644)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _build_reference(seed: int, batch_size: int, seq_len: int) -> tuple[TinyMoETransformer, torch.Tensor]:
    torch.manual_seed(seed)
    tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len))
    model = TinyMoETransformer(**_MODEL_KWARGS)
    return model, tokens


def _rule_by_param_name(model: TinyMoETransformerTpSp) -> dict[str, str]:
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
                source = param.grad if grads else param.detach()
                if source is None:
                    source = torch.zeros_like(param)
                tensors[full_name] = source.detach().clone().cpu()
    return tensors


def _gather_object_dict(local: dict[str, torch.Tensor], group: dist.ProcessGroup | None) -> dict[str, torch.Tensor]:
    if group is None:
        return local
    gathered: list[dict[str, torch.Tensor]] = [None for _ in range(dist.get_world_size(group))]  # type: ignore[list-item]
    dist.all_gather_object(gathered, local, group=group)
    merged: dict[str, torch.Tensor] = {}
    for shard in gathered:
        merged.update(shard)
    return merged


def _logical_named_tensors(
    model: torch.nn.Module,
    shard_rules: dict[str, str],
    tp_group: dist.ProcessGroup | None,
    pp_group: dist.ProcessGroup | None,
    ep_group: dist.ProcessGroup | None,
    *,
    grads: bool,
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    shared_tensors: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if ".local_experts." in name:
            continue
        source = param.grad if grads else param.detach()
        if source is None:
            source = torch.zeros_like(param)
        shared_tensors[name] = _logical_tensor(name, source.detach(), shard_rules, tp_group).cpu()
    tensors = {name: tensor.to(device) for name, tensor in _gather_object_dict(shared_tensors, pp_group).items()}

    expert_tensors = _gather_object_dict(_runtime_local_expert_tensors(model, grads=grads), ep_group)
    expert_tensors = _gather_object_dict(expert_tensors, pp_group)
    for name, tensor in expert_tensors.items():
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


def _reduce_loss(loss: torch.Tensor, core: RuntimeCore) -> torch.Tensor:
    reduced = loss.detach().clone()
    cp_group = core.get_group(MeshAxis.CP)
    dp_group = core.get_group(MeshAxis.DP)
    if cp_group is not None:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=cp_group)
    if dp_group is not None:
        dist.all_reduce(reduced, op=dist.ReduceOp.AVG, group=dp_group)
    return reduced


def _build_zero3() -> Zero3Plugin:
    return Zero3Plugin(
        wrap_cls=_ZERO3_WRAP_CLS,
    )


def _make_baseline_core(reference_model: TinyMoETransformer, args: argparse.Namespace) -> RuntimeCore:
    model = TinyMoETransformerTpSp(**_MODEL_KWARGS)
    model.load_state_dict(reference_model.state_dict())
    return RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=args.pp_size, cp=args.cp_size, ep=args.ep_size),
        plan=ParallelPlan(zero_stage=3, pp_schedule=PipelineScheduleConfig(microbatches=args.pp_microbatches)),
        model=model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=[
            TensorParallelPlugin(),
            SequenceParallelPlugin(),
            ContextParallelPlugin(),
            ExpertParallelPlugin(),
            _build_zero3(),
        ],
    )


def _make_runtime_core(reference_model: TinyMoETransformer, args: argparse.Namespace) -> RuntimeCore:
    model = TinyMoETransformerTpSp(**_MODEL_KWARGS)
    model.load_state_dict(reference_model.state_dict())
    return RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=args.pp_size, cp=args.cp_size, ep=args.ep_size),
        plan=ParallelPlan(zero_stage=3, pp_schedule=PipelineScheduleConfig(microbatches=args.pp_microbatches)),
        model=model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=[
            TensorParallelPlugin(),
            SequenceParallelPlugin(),
            ContextParallelPlugin(),
            PipelineParallelPlugin(schedule=args.pp_schedule),
            ExpertParallelPlugin(),
            _build_zero3(),
        ],
    )


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )
    if args.world_size != args.dp_size * args.pp_size * args.cp_size * args.tp_size:
        raise ValueError("EP full stack expects world_size == dp_size * pp_size * cp_size * tp_size")
    if args.batch_size % args.dp_size != 0:
        raise ValueError("batch size must be divisible by dp size")
    if args.seq_len % args.cp_size != 0:
        raise ValueError("seq_len must be divisible by cp size")

    reference_model, tokens = _build_reference(args.seed, args.batch_size, args.seq_len)

    baseline_core = _make_baseline_core(reference_model, args)
    runtime_core = _make_runtime_core(reference_model, args)
    baseline_core.setup()
    runtime_core.setup()
    baseline_core.model.train()
    runtime_core.model.train()

    dp_idx, _, _, _ = runtime_core.mesh.rank_coordinates(rank)
    local_batch_size = args.batch_size // args.dp_size
    local_tokens = tokens.narrow(0, dp_idx * local_batch_size, local_batch_size).contiguous()

    baseline_loss, should_step = baseline_core.run_step(causal_lm_batch(local_tokens))
    if not should_step:
        raise AssertionError("baseline EP full stack expected should_step=True")
    runtime_loss, should_step = runtime_core.run_step(causal_lm_batch(local_tokens))
    if not should_step:
        raise AssertionError("runtime EP full stack expected should_step=True")

    reduced_baseline_loss = _reduce_loss(baseline_loss, baseline_core)
    reduced_runtime_loss = _reduce_loss(runtime_loss, runtime_core)

    baseline_core._run_phase(RuntimePhase.PRE_STEP)
    runtime_core._run_phase(RuntimePhase.PRE_STEP)

    baseline_zero3 = next(plugin for plugin in baseline_core.plugins if isinstance(plugin, Zero3Plugin))
    runtime_zero3 = next(plugin for plugin in runtime_core.plugins if isinstance(plugin, Zero3Plugin))
    shard_rules = _rule_by_param_name(TinyMoETransformerTpSp(**_MODEL_KWARGS))
    baseline_tp_group = baseline_core.get_group(MeshAxis.TP)
    runtime_tp_group = runtime_core.get_group(MeshAxis.TP)
    baseline_pp_group = baseline_core.get_group(MeshAxis.PP)
    runtime_pp_group = runtime_core.get_group(MeshAxis.PP)
    baseline_ep_group = baseline_core.get_group(MeshAxis.EP)
    runtime_ep_group = runtime_core.get_group(MeshAxis.EP)

    baseline_grads = _logical_named_tensors(
        baseline_core.model, shard_rules, baseline_tp_group, baseline_pp_group, baseline_ep_group, grads=True
    )
    runtime_grads = _logical_named_tensors(
        runtime_core.model, shard_rules, runtime_tp_group, runtime_pp_group, runtime_ep_group, grads=True
    )
    grad_name, grad_diff = _max_diff(baseline_grads, runtime_grads)

    baseline_core.step_optimizer()
    runtime_core.step_optimizer()

    baseline_zero3.materialize_model()
    runtime_zero3.materialize_model()
    baseline_params = _logical_named_tensors(
        baseline_core.model, shard_rules, baseline_tp_group, baseline_pp_group, baseline_ep_group, grads=False
    )
    runtime_params = _logical_named_tensors(
        runtime_core.model, shard_rules, runtime_tp_group, runtime_pp_group, runtime_ep_group, grads=False
    )
    step_name, step_diff = _max_diff(baseline_params, runtime_params)
    baseline_zero3.reshard_model()
    runtime_zero3.reshard_model()

    if rank == 0:
        loss_diff = abs(reduced_baseline_loss.item() - reduced_runtime_loss.item())
        print(f"Baseline loss                 : {reduced_baseline_loss.item():.6f}")
        print(f"RuntimeCore EP full stack     : {reduced_runtime_loss.item():.6f}")
        print(f"Diff                          : {loss_diff:.2e}  (atol={_LOSS_ATOL:.2e})")
        print(f"Grad diff                     : {grad_diff:.2e}  ({grad_name}, atol={_GRAD_ATOL:.2e})")
        print(f"Step diff                     : {step_diff:.2e}  ({step_name}, atol={_STEP_ATOL:.2e})")
        if loss_diff > _LOSS_ATOL:
            raise AssertionError(f"EP full stack loss mismatch: diff={loss_diff:.2e}")
        if grad_diff > _GRAD_ATOL:
            raise AssertionError(f"EP full stack grad mismatch: {grad_name} diff={grad_diff:.2e}")
        if step_diff > _STEP_ATOL:
            raise AssertionError(f"EP full stack step mismatch: {step_name} diff={step_diff:.2e}")
        print("PASS")

    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
