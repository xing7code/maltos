"""Full-stack PP+CP+TP+SP+EP+ZeRO checkpoint/resume tests.

--grad-accum-steps 1  step-boundary checkpoint (default)
--grad-accum-steps 2  mid-step checkpoint
"""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distributed_test_utils import (
    max_diff as _max_diff,
    moe_named_tensors as _moe_named_tensors,
    reduce_loss as _reduce_loss,
    rule_by_param_name as _rule_by_param_name,
    supports_bf16_autocast as _supports_bf16_autocast,
)
from helpers import causal_lm_batch
from models import TinyMoETransformer, TinyMoETransformerTpSp
from models.tiny_transformer import RmsNorm
from parallel import ContextParallelAttentionCoreType, ParallelPlan
from parallel.schedule import PipelineScheduleConfig
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.plugins.cp import ContextParallelPlugin
from runtime.plugins.ep import ExpertParallelPlugin
from runtime.plugins.grad_clip import GradClipPlugin
from runtime.plugins.pp import PipelineParallelPlugin
from runtime.plugins.precision import PrecisionPlugin
from runtime.plugins.sp import SequenceParallelPlugin
from runtime.plugins.tp import TensorParallelPlugin
from runtime.plugins.zero1 import Zero1Plugin
from runtime.plugins.zero2 import Zero2Plugin
from runtime.plugins.zero3 import Zero3Plugin
from state import load_sharded_checkpoint, save_sharded_checkpoint


_MODEL_KWARGS = dict(
    dim=64, n_heads=4, n_kv_heads=4, hidden_size=128, eps=1e-5,
    n_layers=4, vocab_size=256, max_seq_len=64, num_experts=8,
)
_LOSS_ATOL = 5e-2
_STEP_ATOL = 5e-2
_LR = 1e-2
_ZERO3_WRAP_CLS = {torch.nn.Linear, torch.nn.Embedding, RmsNorm}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--pp-microbatches", type=int, default=2)
    parser.add_argument("--pp-schedule", choices=("afab", "1f1b"), default="afab")
    parser.add_argument("--grad-accum-steps", type=int, choices=(1, 2), default=1)
    parser.add_argument("--cp-attn-core", choices=("all_gather_kv", "ring"), default="all_gather_kv")
    parser.add_argument("--no-reuse-tp-for-ep", dest="reuse_tp_for_ep", action="store_false")
    parser.add_argument("--no-reuse-cp-for-ep", dest="reuse_cp_for_ep", action="store_false")
    parser.set_defaults(reuse_tp_for_ep=True, reuse_cp_for_ep=True)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29635)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--zero-stage", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--disable-precision", action="store_true")
    parser.add_argument("--disable-grad-clip", action="store_true")
    return parser.parse_args()

def _build_reference(
    seed: int, batch_size: int, seq_len: int,
) -> tuple[TinyMoETransformer, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    return (
        TinyMoETransformer(**_MODEL_KWARGS),
        torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len)),
        torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len)),
    )

def _check_bf16_metadata(cont: RuntimeCore, rest: RuntimeCore) -> None:
    if cont.state.metadata.get("loss_scale") is not None:
        raise AssertionError("bf16 path should keep loss_scale=None")
    if rest.state.metadata.get("loss_scale") is not None:
        raise AssertionError("bf16 resume path should keep loss_scale=None")
    if cont.state.metadata.get("overflow") is not False:
        raise AssertionError("bf16 path should keep overflow=False")
    if rest.state.metadata.get("overflow") is not False:
        raise AssertionError("bf16 resume path should keep overflow=False")


def _build_runtime(
    model: TinyMoETransformerTpSp, args: argparse.Namespace, device: torch.device | None = None,
) -> tuple[RuntimeCore, Zero1Plugin | Zero2Plugin | Zero3Plugin | None]:
    zero_plugin: Zero1Plugin | Zero2Plugin | Zero3Plugin | None = None
    if args.zero_stage == 1:
        zero_plugin = Zero1Plugin(bucket_mb_size=0)
    elif args.zero_stage == 2:
        zero_plugin = Zero2Plugin(bucket_mb_size=0)
    elif args.zero_stage == 3:
        zero_plugin = Zero3Plugin(wrap_cls=_ZERO3_WRAP_CLS)
    plugins = []
    if args.tp_size > 1:
        plugins += [TensorParallelPlugin(), SequenceParallelPlugin()]
    if args.cp_size > 1:
        plugins.append(ContextParallelPlugin())
    if args.pp_size > 1:
        plugins.append(PipelineParallelPlugin(schedule=args.pp_schedule))
    plugins.append(ExpertParallelPlugin())
    if zero_plugin is not None:
        plugins.append(zero_plugin)
    if not args.disable_precision:
        plugins.append(PrecisionPlugin(compute_dtype=torch.bfloat16))
    if not args.disable_grad_clip:
        plugins.append(GradClipPlugin(max_norm=1.0))
    core = RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=args.pp_size, cp=args.cp_size, ep=args.ep_size),
        plan=ParallelPlan(
            zero_stage=args.zero_stage,
            pp_schedule=PipelineScheduleConfig(microbatches=args.pp_microbatches),
            cp_attn_core=ContextParallelAttentionCoreType(args.cp_attn_core),
            reuse_tp_for_ep=args.reuse_tp_for_ep,
            reuse_cp_for_ep=args.reuse_cp_for_ep,
        ),
        model=model,
        grad_accum_steps=args.grad_accum_steps,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=plugins,
        device=device,
    )
    core.setup()
    return core, zero_plugin


def _compare_params(
    cont_core: RuntimeCore,
    rest_core: RuntimeCore,
    cont_model: TinyMoETransformerTpSp,
    device: torch.device | None,
) -> tuple[str, float]:
    shard_rules = _rule_by_param_name(cont_model)
    tp_group = cont_core.get_group(MeshAxis.TP)
    pp_group = cont_core.get_group(MeshAxis.PP)
    ep_group = cont_core.get_group(MeshAxis.EP)
    cont_params = _moe_named_tensors(
        cont_core.model,
        shard_rules,
        tp_group,
        ep_group,
        pp_group=pp_group,
        pp_partitioned=pp_group is not None,
        device=device,
    )
    rest_params = _moe_named_tensors(
        rest_core.model,
        shard_rules,
        tp_group,
        ep_group,
        pp_group=pp_group,
        pp_partitioned=pp_group is not None,
        device=device,
    )
    return _max_diff(cont_params, rest_params)


def _run_step_resume(rank: int, args: argparse.Namespace, device: torch.device | None) -> None:
    reference_model, tokens_a, tokens_b = _build_reference(args.seed, args.global_batch_size, args.seq_len)

    cont_model = TinyMoETransformerTpSp(**_MODEL_KWARGS)
    cont_model.load_state_dict(reference_model.state_dict())
    cont_core, cont_zero = _build_runtime(cont_model, args, device)
    rest_core: RuntimeCore | None = None
    try:
        dist.barrier()

        dp_idx, _, _, _ = cont_core.mesh.rank_coordinates(rank)
        local_bs = args.global_batch_size // args.dp_size
        local_a = tokens_a.narrow(0, dp_idx * local_bs, local_bs).contiguous()
        local_b = tokens_b.narrow(0, dp_idx * local_bs, local_bs).contiguous()

        _, should_step = cont_core.run_step(causal_lm_batch(local_a))
        if not should_step:
            raise AssertionError("EP step-resume: expected should_step=True for grad_accum_steps=1")
        cont_core.step_optimizer()
        dist.barrier()
        save_sharded_checkpoint(cont_core.state_manager, args.checkpoint_dir)
        dist.barrier()
        cont_loss, should_step = cont_core.run_step(causal_lm_batch(local_b))
        if not should_step:
            raise AssertionError("EP step-resume: second step should step optimizer")
        cont_core.step_optimizer()
        dist.barrier()

        rest_model = TinyMoETransformerTpSp(**_MODEL_KWARGS)
        rest_model.load_state_dict(reference_model.state_dict())
        rest_core, rest_zero = _build_runtime(rest_model, args, device)
        load_sharded_checkpoint(rest_core.state_manager, args.checkpoint_dir)
        dist.barrier()
        rest_loss, should_step = rest_core.run_step(causal_lm_batch(local_b))
        if not should_step:
            raise AssertionError("EP step-resume: restored second step should step optimizer")
        rest_core.step_optimizer()
        dist.barrier()

        cont_loss_r = _reduce_loss(cont_loss, cont_core)
        rest_loss_r = _reduce_loss(rest_loss, rest_core)
        if isinstance(cont_zero, Zero3Plugin):
            cont_zero.materialize_model()
        if isinstance(rest_zero, Zero3Plugin):
            rest_zero.materialize_model()
        param_name, param_diff = _compare_params(cont_core, rest_core, cont_model, device)

        if rank == 0:
            loss_diff = abs(cont_loss_r.item() - rest_loss_r.item())
            print(f"Checkpoint dir    : {args.checkpoint_dir}")
            print(f"Resume loss diff  : {loss_diff:.2e}  (atol={_LOSS_ATOL:.2e})")
            print(f"Resume param diff : {param_diff:.2e}  ({param_name}, atol={_STEP_ATOL:.2e})")
            if loss_diff > _LOSS_ATOL:
                raise AssertionError(f"EP step-resume loss mismatch: diff={loss_diff:.2e}")
            if param_diff > _STEP_ATOL:
                raise AssertionError(f"EP step-resume param mismatch: {param_name} diff={param_diff:.2e}")
            _check_bf16_metadata(cont_core, rest_core)
            print("PASS")
    finally:
        cont_core.close()
        if rest_core is not None:
            rest_core.close()


def _run_midstep_resume(rank: int, args: argparse.Namespace, device: torch.device | None) -> None:
    reference_model, tokens_a, tokens_b = _build_reference(args.seed, args.global_batch_size, args.seq_len)

    cont_model = TinyMoETransformerTpSp(**_MODEL_KWARGS)
    cont_model.load_state_dict(reference_model.state_dict())
    cont_core, cont_zero = _build_runtime(cont_model, args, device)
    rest_core: RuntimeCore | None = None
    try:
        dp_idx, _, _, _ = cont_core.mesh.rank_coordinates(rank)
        local_bs = args.global_batch_size // args.dp_size
        local_a = tokens_a.narrow(0, dp_idx * local_bs, local_bs).contiguous()
        local_b = tokens_b.narrow(0, dp_idx * local_bs, local_bs).contiguous()
        if local_a.size(0) != 4 or local_b.size(0) != 4:
            raise ValueError("EP mid-step resume expects local_batch_size == grad_accum_steps * pp_microbatches == 4")

        _, should_step = cont_core.run_step(causal_lm_batch(local_a[0:2]))
        if cont_core.state.step != 0 or cont_core.state.step_context.microbatch_idx != 1:
            raise AssertionError("EP midstep: continuous core must be at mid-step state before checkpoint")
        if should_step is not False:
            raise AssertionError("EP midstep: first microbatch should not step optimizer")
        save_sharded_checkpoint(cont_core.state_manager, args.checkpoint_dir)
        la1_cont, should_step = cont_core.run_step(causal_lm_batch(local_a[2:4]))
        if should_step is not True:
            raise AssertionError("EP midstep: second microbatch should step optimizer")
        cont_core.step_optimizer()
        lb0_cont, _ = cont_core.run_step(causal_lm_batch(local_b[0:2]))
        lb1_cont, should_step = cont_core.run_step(causal_lm_batch(local_b[2:4]))
        if should_step is not True:
            raise AssertionError("EP midstep: step B boundary should step optimizer")
        cont_core.step_optimizer()

        rest_model = TinyMoETransformerTpSp(**_MODEL_KWARGS)
        rest_model.load_state_dict(reference_model.state_dict())
        rest_core, rest_zero = _build_runtime(rest_model, args, device)
        load_sharded_checkpoint(rest_core.state_manager, args.checkpoint_dir)
        if rest_core.state.step != 0 or rest_core.state.step_context.microbatch_idx != 1:
            raise AssertionError("EP midstep: restored core must recover mid-step state")
        la1_rest, should_step = rest_core.run_step(causal_lm_batch(local_a[2:4]))
        if should_step is not True:
            raise AssertionError("EP midstep: restored step A boundary should step optimizer")
        rest_core.step_optimizer()
        lb0_rest, should_step = rest_core.run_step(causal_lm_batch(local_b[0:2]))
        if should_step is not False:
            raise AssertionError("EP midstep: restored step B first microbatch should not step optimizer")
        lb1_rest, should_step = rest_core.run_step(causal_lm_batch(local_b[2:4]))
        if should_step is not True:
            raise AssertionError("EP midstep: restored step B boundary should step optimizer")
        rest_core.step_optimizer()

        loss_pairs = [
            ("A1", _reduce_loss(la1_cont, cont_core), _reduce_loss(la1_rest, rest_core)),
            ("B0", _reduce_loss(lb0_cont, cont_core), _reduce_loss(lb0_rest, rest_core)),
            ("B1", _reduce_loss(lb1_cont, cont_core), _reduce_loss(lb1_rest, rest_core)),
        ]
        loss_diffs = [(tag, abs(lhs.item() - rhs.item())) for tag, lhs, rhs in loss_pairs]

        if isinstance(cont_zero, Zero3Plugin):
            cont_zero.materialize_model()
        if isinstance(rest_zero, Zero3Plugin):
            rest_zero.materialize_model()
        param_name, param_diff = _compare_params(cont_core, rest_core, cont_model, device)

        if rank == 0:
            worst_tag, worst_loss = max(loss_diffs, key=lambda x: x[1])
            print(f"Checkpoint dir      : {args.checkpoint_dir}")
            print(f"EP mid-step loss    : {worst_loss:.2e}  ({worst_tag}, atol={_LOSS_ATOL:.2e})")
            print(f"EP mid-step params  : {param_diff:.2e}  ({param_name}, atol={_STEP_ATOL:.2e})")
            if worst_loss > _LOSS_ATOL:
                raise AssertionError(f"EP mid-step resume loss mismatch: tag={worst_tag}, diff={worst_loss:.2e}")
            if param_diff > _STEP_ATOL:
                raise AssertionError(f"EP mid-step resume param mismatch: {param_name} diff={param_diff:.2e}")
            _check_bf16_metadata(cont_core, rest_core)
            print("PASS")
    finally:
        cont_core.close()
        if rest_core is not None:
            rest_core.close()


def _validate_args(args: argparse.Namespace) -> None:
    if args.world_size != args.dp_size * args.pp_size * args.cp_size * args.tp_size:
        raise ValueError("world size must equal dp_size * pp_size * cp_size * tp_size")
    if args.global_batch_size % args.dp_size != 0:
        raise ValueError("global batch size must be divisible by dp size")
    if args.seq_len % args.cp_size != 0:
        raise ValueError("seq_len must be divisible by cp size")


def run_case(rank: int, args: argparse.Namespace, device: torch.device | None = None) -> None:
    _validate_args(args)
    if not _supports_bf16_autocast():
        if rank == 0:
            print("SKIP: bf16 autocast is not supported on this runtime")
        return
    if args.grad_accum_steps == 1:
        _run_step_resume(rank, args, device)
    else:
        _run_midstep_resume(rank, args, device)


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
    if args.checkpoint_dir is None:
        args.checkpoint_dir = tempfile.mkdtemp(prefix="ep_full_stack_resume_")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
