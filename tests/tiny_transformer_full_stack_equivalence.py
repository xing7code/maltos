"""Equivalence test for the full TinyTransformer stack.

Current stack under test:
  - DP=2
  - PP=2
  - CP=optional
  - TP=2
  - SP
  - ZeRO1/2/3

The baseline removes PP but keeps the rest of the stack so we can isolate
pipeline-runtime correctness while preserving the same CP/TP/SP/ZeRO semantics.

Usage:
  PYTHONPATH=. .venv/bin/python tests/tiny_transformer_full_stack_equivalence.py \
    --world-size 8 \
    --dp-size 2 \
    --pp-size 2 \
    --tp-size 2
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from helpers import causal_lm_batch
from models import TinyTransformer, TinyTransformerTpSp
from models.tiny_transformer import RmsNorm
from parallel import ParallelPlan
from parallel.schedule import PipelineScheduleConfig
from parallel.specs import TpSpShardAxis
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.plugins.cp import ContextParallelPlugin
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

_ZERO3_WRAP_CLS = {torch.nn.Linear, torch.nn.Embedding, RmsNorm}
_LOSS_ATOL = 1e-3
_GRAD_ATOL = 1e-5
_STEP_ATOL = 1e-5
_LR = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--pp-microbatches", type=int, default=2)
    parser.add_argument("--pp-schedule", choices=("afab", "1f1b"), default="afab")
    parser.add_argument("--zero-stage", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29569)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero3-disable-prefetch", action="store_true")
    return parser.parse_args()


def _build_reference(seed: int, batch_size: int, seq_len: int) -> tuple[TinyTransformer, torch.Tensor]:
    torch.manual_seed(seed)
    tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len))
    model = TinyTransformer(**_MODEL_KWARGS)
    return model, tokens


def _rule_by_param_name(model) -> dict[str, str]:
    rules = {}
    for rule in model.tpsp_parallelize_spec().rules:
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


def _logical_named_zero_grads(
    core: RuntimeCore,
    zero_plugin: Zero1Plugin | Zero2Plugin | Zero3Plugin,
    shard_rules: dict[str, str],
    tp_group: dist.ProcessGroup | None,
) -> dict[str, torch.Tensor]:
    dp_group = core.get_group(MeshAxis.DP)
    if dp_group is None:
        raise ValueError("full stack ZeRO grad materialization requires DP group")
    name_by_param = {id(param): name for name, param in core.model.named_parameters()}
    logical_grads: dict[str, torch.Tensor] = {}
    for bucket in zero_plugin.buckets:
        local_grad = bucket.local_param.grad
        if local_grad is None:
            continue
        full_bucket_grad = torch.cat(_all_gather_tensor(local_grad.detach(), dp_group), dim=0)
        offset = 0
        logical_names = getattr(bucket, "logical_names", None)
        param_shapes = getattr(bucket, "param_shapes", None)
        param_numels = getattr(bucket, "param_numels", None)
        if logical_names is not None and param_shapes is not None and param_numels is not None:
            for name, numel, shape in zip(logical_names, param_numels, param_shapes):
                grad = full_bucket_grad[offset : offset + numel].view(shape)
                logical_grads[name] = _logical_tensor(name, grad, shard_rules, tp_group)
                offset += numel
            continue
        for param in bucket.params:
            name = name_by_param[id(param)]
            numel = param.numel()
            grad = full_bucket_grad[offset : offset + numel].view_as(param)
            logical_grads[name] = _logical_tensor(name, grad, shard_rules, tp_group)
            offset += numel
    return logical_grads


def _max_diff(lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]) -> tuple[str, float]:
    worst_name = ""
    worst_diff = 0.0
    for name, lhs_tensor in lhs.items():
        diff = (lhs_tensor - rhs[name]).abs().max().item()
        if diff > worst_diff:
            worst_name = name
            worst_diff = diff
    return worst_name, worst_diff


def _get_zero_plugin(core: RuntimeCore) -> Zero1Plugin | Zero2Plugin | Zero3Plugin:
    zero_plugin = next(
        (plugin for plugin in core.plugins if isinstance(plugin, (Zero1Plugin, Zero2Plugin, Zero3Plugin))),
        None,
    )
    if zero_plugin is None:
        raise RuntimeError("full stack equivalence expects a ZeRO plugin")
    return zero_plugin


def _make_zero_plugin(args: argparse.Namespace) -> Zero1Plugin | Zero2Plugin | Zero3Plugin:
    if args.zero_stage == 1:
        return Zero1Plugin(bucket_mb_size=0)
    if args.zero_stage == 2:
        return Zero2Plugin(bucket_mb_size=0)
    return Zero3Plugin(
        wrap_cls=_ZERO3_WRAP_CLS,
        enable_prefetch=not args.zero3_disable_prefetch,
    )


def _make_baseline_core(reference_model: TinyTransformer, args: argparse.Namespace) -> RuntimeCore:
    model = TinyTransformerTpSp(**_MODEL_KWARGS)
    model.load_state_dict(reference_model.state_dict())
    plugins = [
        TensorParallelPlugin(),
        SequenceParallelPlugin(),
    ]
    if args.cp_size > 1:
        plugins.append(ContextParallelPlugin())
    plugins.append(_make_zero_plugin(args))
    return RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=args.pp_size, cp=args.cp_size, ep=1),
        plan=ParallelPlan(zero_stage=args.zero_stage),
        model=model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=plugins,
    )


def _make_runtime_core(reference_model: TinyTransformer, args: argparse.Namespace) -> RuntimeCore:
    model = TinyTransformerTpSp(**_MODEL_KWARGS)
    model.load_state_dict(reference_model.state_dict())
    plugins = [
        TensorParallelPlugin(),
        SequenceParallelPlugin(),
    ]
    if args.cp_size > 1:
        plugins.append(ContextParallelPlugin())
    plugins.append(PipelineParallelPlugin(schedule=args.pp_schedule))
    plugins.append(_make_zero_plugin(args))
    return RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=args.pp_size, cp=args.cp_size, ep=1),
        plan=ParallelPlan(
            pp_schedule=PipelineScheduleConfig(microbatches=args.pp_microbatches),
            zero_stage=args.zero_stage,
        ),
        model=model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=plugins,
    )


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )
    if args.world_size != args.dp_size * args.pp_size * args.cp_size * args.tp_size:
        raise ValueError("full stack equivalence expects world_size == dp_size * pp_size * cp_size * tp_size")
    if args.batch_size % args.dp_size != 0:
        raise ValueError("batch size must be divisible by dp size")
    if args.seq_len % args.cp_size != 0:
        raise ValueError("seq_len must be divisible by cp size")

    reference_model, tokens = _build_reference(args.seed, args.batch_size, args.seq_len)

    baseline_core = _make_baseline_core(reference_model, args)
    baseline_core.setup()
    runtime_core = _make_runtime_core(reference_model, args)
    runtime_core.setup()

    dp_idx = rank // (args.pp_size * args.cp_size * args.tp_size)
    local_batch_size = args.batch_size // args.dp_size
    local_tokens = tokens.narrow(0, dp_idx * local_batch_size, local_batch_size).contiguous()
    batch = causal_lm_batch(local_tokens)

    baseline_loss, baseline_should_step = baseline_core.run_step(batch)
    runtime_loss, runtime_should_step = runtime_core.run_step(batch)
    if not baseline_should_step or not runtime_should_step:
        raise AssertionError("full stack test expects grad_accum_steps=1, should_step must be True")

    baseline_zero = _get_zero_plugin(baseline_core)
    runtime_zero = _get_zero_plugin(runtime_core)

    baseline_tp_group = baseline_core.get_group(MeshAxis.TP)
    runtime_tp_group = runtime_core.get_group(MeshAxis.TP)
    baseline_shard_rules = _rule_by_param_name(baseline_core.model)
    runtime_shard_rules = _rule_by_param_name(runtime_core.model)

    baseline_grads = _logical_named_zero_grads(baseline_core, baseline_zero, baseline_shard_rules, baseline_tp_group)
    runtime_grads = _logical_named_zero_grads(runtime_core, runtime_zero, runtime_shard_rules, runtime_tp_group)
    grad_name, grad_diff = _max_diff(runtime_grads, baseline_grads)

    baseline_core.step_optimizer()
    runtime_core.step_optimizer()

    if isinstance(baseline_zero, Zero3Plugin):
        baseline_zero.materialize_model()
    if isinstance(runtime_zero, Zero3Plugin):
        runtime_zero.materialize_model()
    baseline_params = _logical_named_tensors(baseline_core.model, baseline_shard_rules, baseline_tp_group)
    runtime_params = _logical_named_tensors(runtime_core.model, runtime_shard_rules, runtime_tp_group)
    step_name, step_diff = _max_diff(runtime_params, baseline_params)

    dp_group = runtime_core.get_group(MeshAxis.DP)
    cp_group = runtime_core.get_group(MeshAxis.CP)
    avg_baseline_loss = baseline_loss.detach().clone()
    avg_runtime_loss = runtime_loss.detach().clone()
    if cp_group is not None:
        dist.all_reduce(avg_baseline_loss, op=dist.ReduceOp.SUM, group=cp_group)
        dist.all_reduce(avg_runtime_loss, op=dist.ReduceOp.SUM, group=cp_group)
    if dp_group is not None:
        dist.all_reduce(avg_baseline_loss, op=dist.ReduceOp.AVG, group=dp_group)
        dist.all_reduce(avg_runtime_loss, op=dist.ReduceOp.AVG, group=dp_group)

    loss_diff_tensor = torch.tensor(abs(avg_baseline_loss.item() - avg_runtime_loss.item()), dtype=torch.float64)
    grad_diff_tensor = torch.tensor(grad_diff, dtype=torch.float64)
    step_diff_tensor = torch.tensor(step_diff, dtype=torch.float64)
    dist.all_reduce(loss_diff_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(grad_diff_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(step_diff_tensor, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"Case             : full_stack_pp_cp_tp_sp_zero{args.zero_stage}")
        print(f"PP schedule      : {args.pp_schedule}")
        print(f"Baseline loss    : {avg_baseline_loss.item():.6f}")
        print(f"RuntimeCore loss : {avg_runtime_loss.item():.6f}")
        print(f"Loss diff        : {loss_diff_tensor.item():.2e}  (atol={_LOSS_ATOL:.2e})")
        print(f"Grad diff        : {grad_diff_tensor.item():.2e}  ({grad_name}, atol={_GRAD_ATOL:.2e})")
        print(f"Post-step diff   : {step_diff_tensor.item():.2e}  ({step_name}, atol={_STEP_ATOL:.2e})")
        if loss_diff_tensor.item() > _LOSS_ATOL:
            raise AssertionError(f"full stack loss equivalence failed: diff={loss_diff_tensor.item():.2e}")
        if grad_diff_tensor.item() > _GRAD_ATOL:
            raise AssertionError(
                f"full stack gradient equivalence failed: param={grad_name}, diff={grad_diff_tensor.item():.2e}"
            )
        if step_diff_tensor.item() > _STEP_ATOL:
            raise AssertionError(
                f"full stack one-step equivalence failed: param={step_name}, diff={step_diff_tensor.item():.2e}"
            )
        print("PASS")

    if isinstance(baseline_zero, Zero3Plugin):
        baseline_zero.reshard_model()
    if isinstance(runtime_zero, Zero3Plugin):
        runtime_zero.reshard_model()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
