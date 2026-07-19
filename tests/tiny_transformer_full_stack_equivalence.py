"""Equivalence test for the full TinyTransformer stack.

Current stack under test:
  - DP=2
  - PP=2
  - CP=optional
  - TP=2
  - SP
  - ZeRO0/1/2/3

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
import torch.nn as nn

from distributed_test_utils import (
    max_diff as _max_diff,
    named_tensors as _named_tensors,
    named_zero_bucket_grads as _named_zero_bucket_grads,
    normalize_param_name as _normalize_param_name,
    rule_by_param_name as _rule_by_param_name,
)
from attention_backend_utils import add_attention_backend_arg, resolve_attention_backend
from helpers import causal_lm_batch, packed_causal_lm_batch
from models import OlmoConfig, OlmoForCausalLM, OlmoForCausalLMTpSp, OlmoRMSNorm, TinyTransformer, TinyTransformerTpSp
from models.tiny_transformer import RmsNorm
from parallel import ContextParallelAttentionCoreType, ParallelPlan
from parallel.plan import PipelineScheduleConfig
from runtime.layers.distributed_rmsnorm import DistributedRMSNorm
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.types import RuntimePhase
from runtime.plugins.cp import ContextParallelPlugin
from runtime.plugins.pp import PipelineParallelPlugin
from runtime.plugins.sp import SequenceParallelPlugin
from runtime.plugins.tp import TensorParallelPlugin
from runtime.plugins.zero1 import Zero1Plugin
from runtime.plugins.zero2 import Zero2Plugin
from runtime.plugins.zero3 import Zero3Plugin


_TINY_MODEL_KWARGS = dict(
    dim=64,
    n_heads=4,
    n_kv_heads=4,
    hidden_size=128,
    eps=1e-5,
    n_layers=4,
    vocab_size=256,
    max_seq_len=64,
)
_OLMO_MODEL_KWARGS = dict(
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=4,
    intermediate_size=128,
    num_hidden_layers=4,
    vocab_size=256,
    max_position_embeddings=64,
    rms_norm_eps=1e-6,
)

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
    parser.add_argument("--zero-stage", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--cp-attn-core", choices=("all_gather_kv", "ring"), default="all_gather_kv")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29569)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--packed-batch", action="store_true")
    parser.add_argument("--model", choices=("tiny", "olmo2"), default="tiny")
    add_attention_backend_arg(parser)

    return parser.parse_args()


def _model_vocab_size(model_name: str) -> int:
    if model_name == "tiny":
        return int(_TINY_MODEL_KWARGS["vocab_size"])
    return int(_OLMO_MODEL_KWARGS["vocab_size"])


def _build_model(model_name: str, attention_backend: str, *, parallelized: bool) -> nn.Module:
    if model_name == "tiny":
        cls = TinyTransformerTpSp if parallelized else TinyTransformer
        return cls(**_TINY_MODEL_KWARGS, attention_backend=attention_backend)
    config = OlmoConfig(**_OLMO_MODEL_KWARGS, attention_backend=attention_backend)
    cls = OlmoForCausalLMTpSp if parallelized else OlmoForCausalLM
    return cls(config)


def _zero3_wrap_cls(model_name: str) -> set[type[nn.Module]]:
    if model_name == "tiny":
        return {nn.Linear, nn.Embedding, RmsNorm}
    return {nn.Linear, nn.Embedding, OlmoRMSNorm, DistributedRMSNorm}


def _build_reference(
    seed: int,
    batch_size: int,
    seq_len: int,
    attention_backend: str,
    model_name: str,
) -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(seed)
    tokens = torch.randint(0, _model_vocab_size(model_name), (batch_size, seq_len))
    model = _build_model(model_name, attention_backend, parallelized=False)
    return model, tokens

def _find_zero_plugin(core: RuntimeCore) -> Zero1Plugin | Zero2Plugin | Zero3Plugin | None:
    return next(
        (plugin for plugin in core.plugins if isinstance(plugin, (Zero1Plugin, Zero2Plugin, Zero3Plugin))),
        None,
    )


def _make_zero_plugin(args: argparse.Namespace) -> Zero1Plugin | Zero2Plugin | Zero3Plugin | None:
    if args.zero_stage == 0:
        return None
    if args.zero_stage == 1:
        return Zero1Plugin(bucket_mb_size=0)
    if args.zero_stage == 2:
        return Zero2Plugin(bucket_mb_size=0)
    return Zero3Plugin(wrap_cls=_zero3_wrap_cls(args.model))


def _make_baseline_core(reference_model: nn.Module, args: argparse.Namespace, device: torch.device | None = None) -> RuntimeCore:
    model = _build_model(args.model, args.resolved_attention_backend, parallelized=True)
    model.load_state_dict(reference_model.state_dict())
    plugins = []
    if args.tp_size > 1:
        plugins += [TensorParallelPlugin(), SequenceParallelPlugin()]
    if args.cp_size > 1:
        plugins.append(ContextParallelPlugin())
    zero_plugin = _make_zero_plugin(args)
    if zero_plugin is not None:
        plugins.append(zero_plugin)
    return RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=args.pp_size, cp=args.cp_size, ep=1),
        plan=ParallelPlan(
            cp_attn_core=ContextParallelAttentionCoreType(args.cp_attn_core),
        ),
        model=model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=plugins,
        device=device,
    )


def _make_runtime_core(reference_model: nn.Module, args: argparse.Namespace, device: torch.device | None = None) -> RuntimeCore:
    model = _build_model(args.model, args.resolved_attention_backend, parallelized=True)
    model.load_state_dict(reference_model.state_dict())
    plugins = []
    if args.tp_size > 1:
        plugins += [TensorParallelPlugin(), SequenceParallelPlugin()]
    if args.cp_size > 1:
        plugins.append(ContextParallelPlugin())
    if args.pp_size > 1:
        plugins.append(PipelineParallelPlugin(schedule=args.pp_schedule))
    zero_plugin = _make_zero_plugin(args)
    if zero_plugin is not None:
        plugins.append(zero_plugin)
    return RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=args.pp_size, cp=args.cp_size, ep=1),
        plan=ParallelPlan(
            pp_schedule=PipelineScheduleConfig(microbatches=args.pp_microbatches),
            cp_attn_core=ContextParallelAttentionCoreType(args.cp_attn_core),
        ),
        model=model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=plugins,
        device=device,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.world_size != args.dp_size * args.pp_size * args.cp_size * args.tp_size:
        raise ValueError("full stack equivalence expects world_size == dp_size * pp_size * cp_size * tp_size")
    if args.batch_size % args.dp_size != 0:
        raise ValueError("batch size must be divisible by dp size")
    if args.seq_len % args.cp_size != 0:
        raise ValueError("seq_len must be divisible by cp size")


def _make_batch(input_ids: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
    if args.packed_batch:
        return packed_causal_lm_batch(input_ids)
    return causal_lm_batch(input_ids)


def run_case(rank: int, args: argparse.Namespace, device: torch.device | None = None) -> None:
    _validate_args(args)
    args.resolved_attention_backend = resolve_attention_backend(
        args.attention_backend,
        dist_backend=args.backend,
        device=device,
        require_dense_block=args.cp_size > 1 and args.cp_attn_core == "ring",
        allow_flash=not (args.packed_batch and args.cp_size > 1 and args.cp_attn_core == "ring"),
    )
    reference_model, tokens = _build_reference(
        args.seed,
        args.batch_size,
        args.seq_len,
        args.resolved_attention_backend,
        args.model,
    )

    baseline_core = _make_baseline_core(reference_model, args, device)
    baseline_core.setup()
    runtime_core = _make_runtime_core(reference_model, args, device)
    runtime_core.setup()

    try:
        dp_idx = rank // (args.pp_size * args.cp_size * args.tp_size)
        local_batch_size = args.batch_size // args.dp_size
        local_tokens = tokens.narrow(0, dp_idx * local_batch_size, local_batch_size).contiguous()
        batch = _make_batch(local_tokens, args)

        baseline_loss, baseline_should_step = baseline_core.run_step(batch)
        runtime_loss, runtime_should_step = runtime_core.run_step(batch)
        if not baseline_should_step or not runtime_should_step:
            raise AssertionError("full stack test expects grad_accum_steps=1, should_step must be True")

        baseline_zero = _find_zero_plugin(baseline_core)
        runtime_zero = _find_zero_plugin(runtime_core)

        baseline_tp_group = baseline_core.get_group(MeshAxis.TP)
        runtime_tp_group = runtime_core.get_group(MeshAxis.TP)
        baseline_shard_rules = _rule_by_param_name(baseline_core.model)
        runtime_shard_rules = _rule_by_param_name(runtime_core.model)

        baseline_core._run_step_phase(RuntimePhase.PRE_STEP)
        runtime_core._run_step_phase(RuntimePhase.PRE_STEP)
        baseline_grads = (
            _named_tensors(
                baseline_core.model,
                baseline_shard_rules,
                baseline_tp_group,
                grads=True,
                normalize_name=_normalize_param_name,
            )
            if baseline_zero is None
            else _named_zero_bucket_grads(
                baseline_core.model,
                baseline_zero,
                baseline_shard_rules,
                baseline_tp_group,
                normalize_name=_normalize_param_name,
            )
        )
        runtime_grads = (
            _named_tensors(
                runtime_core.model,
                runtime_shard_rules,
                runtime_tp_group,
                grads=True,
                normalize_name=_normalize_param_name,
            )
            if runtime_zero is None
            else _named_zero_bucket_grads(
                runtime_core.model,
                runtime_zero,
                runtime_shard_rules,
                runtime_tp_group,
                normalize_name=_normalize_param_name,
            )
        )
        grad_name, grad_diff = _max_diff(runtime_grads, baseline_grads)

        baseline_core.step_optimizer()
        runtime_core.step_optimizer()

        if isinstance(baseline_zero, Zero3Plugin):
            baseline_zero.materialize_model()
        if isinstance(runtime_zero, Zero3Plugin):
            runtime_zero.materialize_model()
        baseline_params = _named_tensors(
            baseline_core.model,
            baseline_shard_rules,
            baseline_tp_group,
            normalize_name=_normalize_param_name,
        )
        runtime_params = _named_tensors(
            runtime_core.model,
            runtime_shard_rules,
            runtime_tp_group,
            normalize_name=_normalize_param_name,
        )
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

        loss_diff_tensor = torch.tensor(abs(avg_baseline_loss.item() - avg_runtime_loss.item()), dtype=torch.float64, device=device)
        grad_diff_tensor = torch.tensor(grad_diff, dtype=torch.float64, device=device)
        step_diff_tensor = torch.tensor(step_diff, dtype=torch.float64, device=device)
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
    finally:
        baseline_core.close()
        runtime_core.close()


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    device: torch.device | None = None
    if args.backend == "nccl":
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
        device_id=device,
    )
    try:
        run_case(rank, args, device)
    finally:
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
