"""Mid-step resume test for full-stack PP+CP+TP+SP+ZeRO3 with grad accumulation."""

from __future__ import annotations

import argparse
import os
import tempfile

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
from runtime.plugins.grad_clip import GradClipPlugin
from runtime.plugins.pp import PipelineParallelPlugin
from runtime.plugins.precision import PrecisionPlugin
from runtime.plugins.sp import SequenceParallelPlugin
from runtime.plugins.tp import TensorParallelPlugin
from runtime.plugins.zero3 import Zero3Plugin
from state import load_sharded_checkpoint, save_sharded_checkpoint


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
_LOSS_ATOL = 5e-2
_STEP_ATOL = 5e-2
_LR = 1e-2
_ZERO3_WRAP_CLS = {torch.nn.Linear, torch.nn.Embedding, RmsNorm}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--cp-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--pp-microbatches", type=int, default=2)
    parser.add_argument("--pp-schedule", choices=("afab", "1f1b"), default="afab")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29621)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def _supports_bf16_autocast() -> bool:
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            x = torch.randn(8, 8)
            _ = x @ x
        return True
    except Exception:
        return False


def _build_reference(seed: int, batch_size: int, seq_len: int) -> tuple[TinyTransformer, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    tokens_a = torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len))
    tokens_b = torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len))
    model = TinyTransformer(**_MODEL_KWARGS)
    return model, tokens_a, tokens_b


def _all_gather_tensor(tensor: torch.Tensor, group: dist.ProcessGroup) -> list[torch.Tensor]:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return gathered


def _rule_by_param_name(model: TinyTransformerTpSp) -> dict[str, str]:
    rules = {}
    for rule in model.tpsp_parallelize_spec().rules:
        if rule.shard_axis in (TpSpShardAxis.PARAM_OUT, TpSpShardAxis.PARAM_IN):
            rules[f"{rule.module_path}.weight"] = rule.shard_axis
            rules[f"{rule.module_path}.bias"] = rule.shard_axis
    return rules


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
    worst_name = next(iter(lhs), "")
    worst_diff = 0.0
    for name, lhs_tensor in lhs.items():
        diff = (lhs_tensor - rhs[name]).abs().max().item()
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


def _build_runtime(model: TinyTransformerTpSp, args: argparse.Namespace, device: torch.device | None = None) -> tuple[RuntimeCore, Zero3Plugin]:
    zero3 = Zero3Plugin(
        wrap_cls=_ZERO3_WRAP_CLS,
    )
    core = RuntimeCore(
        mesh=MeshConfig(dp=args.dp_size, tp=args.tp_size, pp=args.pp_size, cp=args.cp_size, ep=1),
        plan=ParallelPlan(
            zero_stage=3,
            pp_schedule=PipelineScheduleConfig(microbatches=args.pp_microbatches),
        ),
        model=model,
        grad_accum_steps=2,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=[
            TensorParallelPlugin(),
            SequenceParallelPlugin(),
            ContextParallelPlugin(),
            PipelineParallelPlugin(schedule=args.pp_schedule),
            zero3,
            PrecisionPlugin(compute_dtype=torch.bfloat16),
            GradClipPlugin(max_norm=1.0),
        ],
        device=device,
    )
    core.setup()
    return core, zero3


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )
    device: torch.device | None = None
    if args.backend == "nccl":
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    if args.world_size != args.dp_size * args.pp_size * args.cp_size * args.tp_size:
        raise ValueError("world size must equal dp_size * pp_size * cp_size * tp_size")
    if args.global_batch_size % args.dp_size != 0:
        raise ValueError("global batch size must be divisible by dp size")
    if args.seq_len % args.cp_size != 0:
        raise ValueError("seq_len must be divisible by cp size")

    if not _supports_bf16_autocast():
        if rank == 0:
            print("SKIP: bf16 autocast is not supported on this runtime")
        dist.destroy_process_group()
        return

    reference_model, tokens_a, tokens_b = _build_reference(args.seed, args.global_batch_size, args.seq_len)

    continuous_model = TinyTransformerTpSp(**_MODEL_KWARGS)
    continuous_model.load_state_dict(reference_model.state_dict())
    continuous_core, continuous_zero3 = _build_runtime(continuous_model, args, device)

    dp_idx = rank // (args.pp_size * args.cp_size * args.tp_size)
    local_batch_size = args.global_batch_size // args.dp_size
    local_a = tokens_a.narrow(0, dp_idx * local_batch_size, local_batch_size).contiguous()
    local_b = tokens_b.narrow(0, dp_idx * local_batch_size, local_batch_size).contiguous()
    if local_a.size(0) != 4 or local_b.size(0) != 4:
        raise ValueError("this test expects local batch size == grad_accum_steps * pp_microbatches == 4")

    loss_a0_cont, should_step = continuous_core.run_step(causal_lm_batch(local_a[0:2]))
    loss_a0_cont = loss_a0_cont.detach()
    if continuous_core.state.step != 0 or continuous_core.state.step_context.microbatch_idx != 1:
        raise AssertionError("continuous core must be at mid-step state before checkpoint")
    if should_step is not False:
        raise AssertionError("continuous first microbatch should not step optimizer")
    save_sharded_checkpoint(continuous_core.state_manager, args.checkpoint_dir)
    loss_a1_cont, should_step = continuous_core.run_step(causal_lm_batch(local_a[2:4]))
    loss_a1_cont = loss_a1_cont.detach()
    if should_step is not True:
        raise AssertionError("continuous second microbatch should step optimizer")
    continuous_core.step_optimizer()
    loss_b0_cont, _ = continuous_core.run_step(causal_lm_batch(local_b[0:2]))
    loss_b0_cont = loss_b0_cont.detach()
    loss_b1_cont, should_step = continuous_core.run_step(causal_lm_batch(local_b[2:4]))
    loss_b1_cont = loss_b1_cont.detach()
    if should_step is not True:
        raise AssertionError("continuous second boundary microbatch should step optimizer")
    continuous_core.step_optimizer()

    restored_model = TinyTransformerTpSp(**_MODEL_KWARGS)
    restored_model.load_state_dict(reference_model.state_dict())
    restored_core, restored_zero3 = _build_runtime(restored_model, args, device)
    load_sharded_checkpoint(restored_core.state_manager, args.checkpoint_dir)
    if restored_core.state.step != 0 or restored_core.state.step_context.microbatch_idx != 1:
        raise AssertionError("restored core must recover mid-step state")
    loss_a1_res, should_step = restored_core.run_step(causal_lm_batch(local_a[2:4]))
    loss_a1_res = loss_a1_res.detach()
    if should_step is not True:
        raise AssertionError("restored boundary microbatch should step optimizer")
    restored_core.step_optimizer()
    loss_b0_res, should_step = restored_core.run_step(causal_lm_batch(local_b[0:2]))
    loss_b0_res = loss_b0_res.detach()
    if should_step is not False:
        raise AssertionError("restored non-boundary microbatch should not step optimizer")
    loss_b1_res, should_step = restored_core.run_step(causal_lm_batch(local_b[2:4]))
    loss_b1_res = loss_b1_res.detach()
    if should_step is not True:
        raise AssertionError("restored boundary microbatch should step optimizer")
    restored_core.step_optimizer()

    loss_pairs = [
        ("A1", _reduce_loss(loss_a1_cont, continuous_core), _reduce_loss(loss_a1_res, restored_core)),
        ("B0", _reduce_loss(loss_b0_cont, continuous_core), _reduce_loss(loss_b0_res, restored_core)),
        ("B1", _reduce_loss(loss_b1_cont, continuous_core), _reduce_loss(loss_b1_res, restored_core)),
    ]
    reduced_loss_diffs = [(tag, abs(lhs.item() - rhs.item())) for tag, lhs, rhs in loss_pairs]

    continuous_zero3.materialize_model()
    restored_zero3.materialize_model()
    shard_rules = _rule_by_param_name(continuous_model)
    tp_group = continuous_core.get_group(MeshAxis.TP)
    continuous_params = _logical_named_tensors(continuous_core.model, shard_rules, tp_group)
    restored_params = _logical_named_tensors(restored_core.model, shard_rules, tp_group)
    param_name, param_diff = _max_diff(continuous_params, restored_params)

    if rank == 0:
        worst_loss_tag, worst_loss_diff = max(reduced_loss_diffs, key=lambda item: item[1])
        print(f"Checkpoint dir      : {args.checkpoint_dir}")
        print(f"Mid-step resume loss: {worst_loss_diff:.2e}  ({worst_loss_tag}, atol={_LOSS_ATOL:.2e})")
        print(f"Mid-step resume param diff: {param_diff:.2e}  ({param_name}, atol={_STEP_ATOL:.2e})")
        if worst_loss_diff > _LOSS_ATOL:
            raise AssertionError(f"mid-step full-stack resume loss mismatch: tag={worst_loss_tag}, diff={worst_loss_diff:.2e}")
        if param_diff > _STEP_ATOL:
            raise AssertionError(f"mid-step full-stack resume param mismatch: {param_name} diff={param_diff:.2e}")
        if continuous_core.state.metadata.get("loss_scale") is not None:
            raise AssertionError("bf16 path should keep loss_scale=None")
        if restored_core.state.metadata.get("loss_scale") is not None:
            raise AssertionError("bf16 resume path should keep loss_scale=None")
        if continuous_core.state.metadata.get("overflow") is not False:
            raise AssertionError("bf16 path should keep overflow=False")
        if restored_core.state.metadata.get("overflow") is not False:
            raise AssertionError("bf16 resume path should keep overflow=False")
        print("PASS")

    continuous_zero3.reshard_model()
    restored_zero3.reshard_model()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = tempfile.mkdtemp(prefix="full_stack_zero3_accum2_midstep_resume_")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
