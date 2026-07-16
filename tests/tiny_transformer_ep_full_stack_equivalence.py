"""Equivalence test for the EP full stack: PP+CP+TP+SP+EP+ZeRO3."""

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
    reduce_loss as _reduce_loss,
    rule_by_param_name as _rule_by_param_name,
)
from helpers import causal_lm_batch, packed_causal_lm_batch
from models import TinyMoETransformer, TinyMoETransformerTpSp
from models.tiny_transformer import RmsNorm
from parallel import ContextParallelAttentionCoreType, ParallelPlan
from parallel.plan import PipelineScheduleConfig
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.types import RuntimePhase
from runtime.plugins.cp import ContextParallelPlugin
from runtime.plugins.ep import ExpertParallelPlugin
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
    num_experts=8,
)

_ZERO3_WRAP_CLS = {torch.nn.Linear, torch.nn.Embedding, RmsNorm}
_LOSS_ATOL = 5e-3
_GRAD_ATOL = 2e-2
_STEP_ATOL = 5e-4
_LR = 1e-2


def _make_zero_plugin(args: argparse.Namespace) -> Zero1Plugin | Zero2Plugin | Zero3Plugin | None:
    if args.zero_stage == 0:
        return None
    if args.zero_stage == 1:
        return Zero1Plugin(bucket_mb_size=0)
    if args.zero_stage == 2:
        return Zero2Plugin(bucket_mb_size=0)
    return Zero3Plugin(wrap_cls=_ZERO3_WRAP_CLS)


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
    parser.add_argument("--zero-stage", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--cp-attn-core", choices=("all_gather_kv", "ring"), default="all_gather_kv")
    parser.add_argument("--no-reuse-tp-for-ep", dest="reuse_tp_for_ep", action="store_false")
    parser.add_argument("--no-reuse-cp-for-ep", dest="reuse_cp_for_ep", action="store_false")
    parser.set_defaults(reuse_tp_for_ep=True, reuse_cp_for_ep=True)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29644)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--packed-batch", action="store_true")
    return parser.parse_args()


def _build_reference(seed: int, batch_size: int, seq_len: int) -> tuple[TinyMoETransformer, torch.Tensor]:
    torch.manual_seed(seed)
    tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len))
    model = TinyMoETransformer(**_MODEL_KWARGS)
    return model, tokens


def _make_baseline_core(reference_model: TinyMoETransformer, args: argparse.Namespace, device: torch.device | None = None) -> RuntimeCore:
    model = TinyMoETransformerTpSp(**_MODEL_KWARGS)
    model.load_state_dict(reference_model.state_dict())
    plugins = []
    if args.tp_size > 1:
        plugins += [TensorParallelPlugin(), SequenceParallelPlugin()]
    if args.cp_size > 1:
        plugins.append(ContextParallelPlugin())
    plugins.append(ExpertParallelPlugin())
    zero_plugin = _make_zero_plugin(args)
    if zero_plugin is not None:
        plugins.append(zero_plugin)
    return RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=args.pp_size, cp=args.cp_size, ep=args.ep_size),
        plan=ParallelPlan(
            pp_schedule=PipelineScheduleConfig(microbatches=args.pp_microbatches),
            cp_attn_core=ContextParallelAttentionCoreType(args.cp_attn_core),
            reuse_tp_for_ep=args.reuse_tp_for_ep,
            reuse_cp_for_ep=args.reuse_cp_for_ep,
        ),
        model=model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=plugins,
        device=device,
    )


def _make_runtime_core(reference_model: TinyMoETransformer, args: argparse.Namespace, device: torch.device | None = None) -> RuntimeCore:
    model = TinyMoETransformerTpSp(**_MODEL_KWARGS)
    model.load_state_dict(reference_model.state_dict())
    plugins = []
    if args.tp_size > 1:
        plugins += [TensorParallelPlugin(), SequenceParallelPlugin()]
    if args.cp_size > 1:
        plugins.append(ContextParallelPlugin())
    if args.pp_size > 1:
        plugins.append(PipelineParallelPlugin(schedule=args.pp_schedule))
    plugins.append(ExpertParallelPlugin())
    zero_plugin = _make_zero_plugin(args)
    if zero_plugin is not None:
        plugins.append(zero_plugin)
    return RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=args.pp_size, cp=args.cp_size, ep=args.ep_size),
        plan=ParallelPlan(
            pp_schedule=PipelineScheduleConfig(microbatches=args.pp_microbatches),
            cp_attn_core=ContextParallelAttentionCoreType(args.cp_attn_core),
            reuse_tp_for_ep=args.reuse_tp_for_ep,
            reuse_cp_for_ep=args.reuse_cp_for_ep,
        ),
        model=model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=plugins,
        device=device,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.world_size != args.dp_size * args.pp_size * args.cp_size * args.tp_size:
        raise ValueError("EP full stack expects world_size == dp_size * pp_size * cp_size * tp_size")
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
    reference_model, tokens = _build_reference(args.seed, args.batch_size, args.seq_len)

    baseline_core = _make_baseline_core(reference_model, args, device)
    runtime_core = _make_runtime_core(reference_model, args, device)
    baseline_core.setup()
    runtime_core.setup()
    baseline_core.model.train()
    runtime_core.model.train()

    try:
        dp_idx, _, _, _ = runtime_core.mesh.rank_coordinates(rank)
        local_batch_size = args.batch_size // args.dp_size
        local_tokens = tokens.narrow(0, dp_idx * local_batch_size, local_batch_size).contiguous()

        baseline_loss, should_step = baseline_core.run_step(_make_batch(local_tokens, args))
        if not should_step:
            raise AssertionError("baseline EP full stack expected should_step=True")
        runtime_loss, should_step = runtime_core.run_step(_make_batch(local_tokens, args))
        if not should_step:
            raise AssertionError("runtime EP full stack expected should_step=True")

        reduced_baseline_loss = _reduce_loss(baseline_loss, baseline_core)
        reduced_runtime_loss = _reduce_loss(runtime_loss, runtime_core)

        baseline_core._run_phase(RuntimePhase.PRE_STEP)
        runtime_core._run_phase(RuntimePhase.PRE_STEP)

        baseline_zero3 = next((plugin for plugin in baseline_core.plugins if isinstance(plugin, Zero3Plugin)), None)
        runtime_zero3 = next((plugin for plugin in runtime_core.plugins if isinstance(plugin, Zero3Plugin)), None)
        baseline_zero23 = next((p for p in baseline_core.plugins if isinstance(p, (Zero2Plugin, Zero3Plugin))), None)
        runtime_zero23 = next((p for p in runtime_core.plugins if isinstance(p, (Zero2Plugin, Zero3Plugin))), None)
        shard_rules = _rule_by_param_name(TinyMoETransformerTpSp(**_MODEL_KWARGS))
        baseline_tp_group = baseline_core.get_group(MeshAxis.TP)
        runtime_tp_group = runtime_core.get_group(MeshAxis.TP)
        baseline_pp_group = baseline_core.get_group(MeshAxis.PP)
        runtime_pp_group = runtime_core.get_group(MeshAxis.PP)
        baseline_ep_group = baseline_core.get_group(MeshAxis.EP)
        runtime_ep_group = runtime_core.get_group(MeshAxis.EP)
        baseline_pp_partitioned = any(isinstance(plugin, PipelineParallelPlugin) for plugin in baseline_core.plugins)
        runtime_pp_partitioned = any(isinstance(plugin, PipelineParallelPlugin) for plugin in runtime_core.plugins)

        if baseline_zero23 is not None:
            baseline_grads = _moe_zero_named_grads(
                baseline_core.model,
                baseline_zero23,
                shard_rules,
                baseline_tp_group,
                baseline_ep_group,
                pp_group=baseline_pp_group,
                pp_partitioned=baseline_pp_partitioned,
            )
            runtime_grads = _moe_zero_named_grads(
                runtime_core.model,
                runtime_zero23,
                shard_rules,
                runtime_tp_group,
                runtime_ep_group,
                pp_group=runtime_pp_group,
                pp_partitioned=runtime_pp_partitioned,
            )
        else:
            baseline_grads = _moe_named_tensors(
                baseline_core.model,
                shard_rules,
                baseline_tp_group,
                baseline_ep_group,
                pp_group=baseline_pp_group,
                grads=True,
                pp_partitioned=baseline_pp_partitioned,
            )
            runtime_grads = _moe_named_tensors(
                runtime_core.model,
                shard_rules,
                runtime_tp_group,
                runtime_ep_group,
                pp_group=runtime_pp_group,
                grads=True,
                pp_partitioned=runtime_pp_partitioned,
            )
        grad_name, grad_diff = _max_diff(baseline_grads, runtime_grads)

        baseline_core.step_optimizer()
        runtime_core.step_optimizer()

        if baseline_zero3 is not None:
            baseline_zero3.materialize_model()
        if runtime_zero3 is not None:
            runtime_zero3.materialize_model()
        baseline_params = _moe_named_tensors(
            baseline_core.model,
            shard_rules,
            baseline_tp_group,
            baseline_ep_group,
            pp_group=baseline_pp_group,
            pp_partitioned=baseline_pp_partitioned,
        )
        runtime_params = _moe_named_tensors(
            runtime_core.model,
            shard_rules,
            runtime_tp_group,
            runtime_ep_group,
            pp_group=runtime_pp_group,
            pp_partitioned=runtime_pp_partitioned,
        )
        step_name, step_diff = _max_diff(baseline_params, runtime_params)
        if baseline_zero3 is not None:
            baseline_zero3.reshard_model()
        if runtime_zero3 is not None:
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
