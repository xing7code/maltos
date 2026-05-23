"""Resume test for TP+SP+ZeRO3+bf16+grad-clip RuntimeCore checkpointing."""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train_system.examples import TinyTransformer, TinyTransformerTpSp
from train_system.parallel import ParallelPlan
from train_system.parallel.specs import TpSpShardAxis
from train_system.runtime import MeshAxis, MeshConfig, RuntimeCore
from train_system.runtime.layers.tp import ColumnParallelLinear, RowParallelLinear
from train_system.runtime.plugins.grad_clip import GradClipPlugin
from train_system.runtime.plugins.precision import PrecisionPlugin
from train_system.runtime.plugins.sp import SequenceParallelPlugin
from train_system.runtime.plugins.tp import TensorParallelPlugin
from train_system.runtime.plugins.zero3 import Zero3Plugin
from train_system.state import load_sharded_checkpoint, save_sharded_checkpoint


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
_LOSS_ATOL = 5e-2
_STEP_ATOL = 5e-2
_LR = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29536)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def _mesh_indices(rank: int, tp_size: int) -> tuple[int, int]:
    return rank // tp_size, rank % tp_size


def _build_reference(seed: int, batch_size: int, seq_len: int) -> tuple[TinyTransformer, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    first_tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len))
    second_tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len))
    model = TinyTransformer(**_MODEL_KWARGS)
    return model, first_tokens, second_tokens


def _local_tokens(full_tokens: torch.Tensor, dp_idx: int, dp_size: int) -> torch.Tensor:
    local_batch_size = full_tokens.size(0) // dp_size
    return full_tokens.narrow(0, dp_idx * local_batch_size, local_batch_size).contiguous()


def _all_gather_tensor(tensor: torch.Tensor, group: dist.ProcessGroup) -> list[torch.Tensor]:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return gathered


def _rule_by_param_name(model: TinyTransformerTpSp) -> dict[str, str]:
    rules = {}
    for rule in model.parallelize_spec().rules:
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
    worst_name = ""
    worst_diff = 0.0
    for name, lhs_tensor in lhs.items():
        diff = (lhs_tensor - rhs[name]).abs().max().item()
        if diff > worst_diff:
            worst_name = name
            worst_diff = diff
    return worst_name, worst_diff


def _build_runtime(model: TinyTransformerTpSp, dp_size: int, tp_size: int) -> tuple[RuntimeCore, Zero3Plugin]:
    zero3 = Zero3Plugin(
        wrap_cls={torch.nn.Linear, ColumnParallelLinear, RowParallelLinear},
        optimizer_cls=torch.optim.SGD,
        lr=_LR,
    )
    core = RuntimeCore(
        mesh=MeshConfig(dp=dp_size, tp=tp_size, pp=1, cp=1, ep=1),
        plan=ParallelPlan(zero_stage=3),
        model=model,
        optimizer=None,
        plugins=[
            TensorParallelPlugin(),
            SequenceParallelPlugin(),
            zero3,
            PrecisionPlugin(compute_dtype=torch.bfloat16),
            GradClipPlugin(max_norm=1.0),
        ],
    )
    core.setup()
    return core, zero3


def _supports_bf16_autocast() -> bool:
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            x = torch.randn(8, 8)
            _ = x @ x
        return True
    except Exception:
        return False


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

    if not _supports_bf16_autocast():
        if rank == 0:
            print("SKIP: bf16 autocast is not supported on this runtime")
        dist.destroy_process_group()
        return

    dp_idx, _ = _mesh_indices(rank, args.tp_size)
    baseline_model, first_tokens, second_tokens = _build_reference(args.seed, args.global_batch_size, args.seq_len)

    continuous_model = TinyTransformerTpSp(**_MODEL_KWARGS)
    continuous_model.load_state_dict(baseline_model.state_dict())
    continuous_core, continuous_zero3 = _build_runtime(continuous_model, args.dp_size, args.tp_size)

    restored_model = TinyTransformerTpSp(**_MODEL_KWARGS)
    restored_model.load_state_dict(baseline_model.state_dict())
    restored_core, restored_zero3 = _build_runtime(restored_model, args.dp_size, args.tp_size)

    first_local = _local_tokens(first_tokens, dp_idx, args.dp_size)
    second_local = _local_tokens(second_tokens, dp_idx, args.dp_size)

    continuous_core.run_train_step((first_local, first_local.clone()))
    save_sharded_checkpoint(continuous_core, args.checkpoint_dir)
    continuous_second_loss = continuous_core.run_train_step((second_local, second_local.clone()))

    load_sharded_checkpoint(restored_core, args.checkpoint_dir)
    restored_second_loss = restored_core.run_train_step((second_local, second_local.clone()))

    dp_group = continuous_core.get_group(MeshAxis.DP)
    continuous_loss = continuous_second_loss.detach().clone()
    restored_loss = restored_second_loss.detach().clone()
    if dp_group is not None:
        dist.all_reduce(continuous_loss, op=dist.ReduceOp.AVG, group=dp_group)
        dist.all_reduce(restored_loss, op=dist.ReduceOp.AVG, group=dp_group)

    continuous_zero3.materialize_model()
    restored_zero3.materialize_model()
    shard_rules = _rule_by_param_name(continuous_model)
    tp_group = continuous_core.get_group(MeshAxis.TP)
    continuous_params = _logical_named_tensors(continuous_core.model, shard_rules, tp_group)
    restored_params = _logical_named_tensors(restored_core.model, shard_rules, tp_group)
    param_name, param_diff = _max_diff(continuous_params, restored_params)

    if rank == 0:
        loss_diff = abs(continuous_loss.item() - restored_loss.item())
        print(f"Checkpoint dir    : {args.checkpoint_dir}")
        print(f"Resume loss diff  : {loss_diff:.2e}  (atol={_LOSS_ATOL:.2e})")
        print(f"Resume param diff : {param_diff:.2e}  ({param_name}, atol={_STEP_ATOL:.2e})")
        if loss_diff > _LOSS_ATOL:
            raise AssertionError(f"loss mismatch after resume: diff={loss_diff:.2e}")
        if param_diff > _STEP_ATOL:
            raise AssertionError(f"param mismatch after resume: {param_name} diff={param_diff:.2e}")
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
        args.checkpoint_dir = tempfile.mkdtemp(prefix="tp_sp_zero3_bf16_clip_resume_")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()

