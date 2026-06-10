"""Equivalence test: EP + CP + ZeRO with DCP/EREP group sharding.

Verifies that combining ExpertParallel, ContextParallel, and ZeRO (1/2/3) on a
MoE model produces correct losses and parameter updates relative to a
single-process reference.

Topology: dp=2, cp=2, ep=2 → world_size=8.  EP uses all DP ranks (ep=dp),
so EREP = (dp/ep)*cp = 1*2 = 2 (CP-parallel expert replicas).  DCP = 4.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from helpers import causal_lm_batch
from models import TinyMoETransformer
from models.tiny_transformer import RmsNorm
from parallel import ParallelPlan
from parallel.context import ContextParallelAttentionCoreType
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.core import RuntimePhase
from runtime.plugins.cp import ContextParallelPlugin
from runtime.plugins.ddp import BucketDataParallelPlugin
from runtime.plugins.ep import ExpertParallelPlugin, _ExpertParallelMoE
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
_STEP_ATOL = 5e-4
_LR = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("ep_cp_zero0", "ep_cp_zero1", "ep_cp_zero2", "ep_cp_zero3"),
        default="ep_cp_zero1",
    )
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--cp-size", type=int, default=2)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29597)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-tp-for-ep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-cp-for-ep", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _tokens_for_dp(seed: int, dp_idx: int, batch_size: int, seq_len: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed + dp_idx)
    return torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len), generator=generator)


def _build_reference(seed: int, dp_size: int, batch_size: int, seq_len: int) -> tuple[TinyMoETransformer, torch.Tensor]:
    torch.manual_seed(seed)
    local_batch = batch_size // dp_size
    tokens = torch.cat([_tokens_for_dp(seed, dp_idx, local_batch, seq_len) for dp_idx in range(dp_size)], dim=0)
    model = TinyMoETransformer(**_MODEL_KWARGS)
    return model, tokens


def _runtime_expert_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, _ExpertParallelMoE):
            continue
        for local_idx, global_idx in enumerate(module.local_expert_ids):
            expert = module.local_experts[local_idx]
            for param_name, param in expert.named_parameters():
                tensors[f"{module_name}.experts.{global_idx}.{param_name}"] = param.detach().clone().cpu()
    return tensors


def _runtime_named_params(
    model: torch.nn.Module,
    ep_group: dist.ProcessGroup | None,
) -> dict[str, torch.Tensor]:
    params: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if ".local_experts." in name:
            continue
        params[name] = param.detach().clone().cpu()
    if ep_group is not None:
        gathered: list[dict[str, torch.Tensor]] = [None for _ in range(dist.get_world_size(ep_group))]  # type: ignore[list-item]
        dist.all_gather_object(gathered, _runtime_expert_tensors(model), group=ep_group)
        for shard in gathered:
            for name, tensor in shard.items():
                params[name] = tensor.cpu()
    return params


def _baseline_named_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone().cpu() for name, param in model.named_parameters()}


def _max_diff(lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]) -> tuple[str, float]:
    worst_name = ""
    worst_diff = 0.0
    for name, lhs_tensor in lhs.items():
        rhs_tensor = rhs[name].to(lhs_tensor.device, lhs_tensor.dtype)
        diff = float((lhs_tensor - rhs_tensor).abs().max().item())
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
    assert args.world_size == args.dp_size * args.cp_size, "world_size must equal dp*cp for this test"
    assert args.seq_len % args.cp_size == 0, "seq_len must be divisible by cp_size"

    mesh = MeshConfig(dp=args.dp_size, tp=1, pp=1, cp=args.cp_size, ep=args.ep_size)
    dp_idx, _, cp_idx, _ = mesh.rank_coordinates(rank)

    # Reference: single-process, full batch, full sequence.
    reference_model, global_tokens = _build_reference(args.seed, args.dp_size, args.batch_size, args.seq_len)
    reference_model.train()
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=_LR)
    reference_optimizer.zero_grad(set_to_none=True)
    reference_loss = reference_model(causal_lm_batch(global_tokens))
    reference_loss.backward()

    # Sharded model.
    sharded_model = TinyMoETransformer(**_MODEL_KWARGS)
    sharded_model.load_state_dict(reference_model.state_dict())
    sharded_model.train()

    cp_attn_core = ContextParallelAttentionCoreType.ALL_GATHER_KV
    zero_stage_map = {"ep_cp_zero0": 0, "ep_cp_zero1": 1, "ep_cp_zero2": 2, "ep_cp_zero3": 3}
    zero_stage = zero_stage_map[args.case]

    zero_plugin: BucketDataParallelPlugin | Zero1Plugin | Zero2Plugin | Zero3Plugin
    if zero_stage == 0:
        zero_plugin = BucketDataParallelPlugin(bucket_mb_size=0)
    elif zero_stage == 1:
        zero_plugin = Zero1Plugin(bucket_mb_size=0)
    elif zero_stage == 2:
        zero_plugin = Zero2Plugin(bucket_mb_size=0)
    else:
        zero_plugin = Zero3Plugin(wrap_cls={torch.nn.Linear, torch.nn.Embedding, RmsNorm})

    plugins = [
        ExpertParallelPlugin(),
        ContextParallelPlugin(),
        zero_plugin,
    ]
    core = RuntimeCore(
        mesh=mesh,
        plan=ParallelPlan(
            zero_stage=zero_stage,
            cp_attn_core=cp_attn_core,
            reuse_tp_for_ep=args.reuse_tp_for_ep,
            reuse_cp_for_ep=args.reuse_cp_for_ep,
        ),
        model=sharded_model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=plugins,
    )
    core.setup()
    core.model.train()

    local_batch_size = args.batch_size // args.dp_size
    local_tokens = _tokens_for_dp(args.seed, dp_idx, local_batch_size, args.seq_len)
    loss, should_step = core.run_step(causal_lm_batch(local_tokens))
    if not should_step:
        raise AssertionError("EP+CP+ZeRO test expected should_step=True")

    # Reduce runtime loss over CP then DP to match reference.
    # Reference uses mean CE; each CP rank holds one slice, sum over CP = full-sequence mean CE.
    runtime_loss = loss.detach().clone()
    cp_group = core.get_group(MeshAxis.CP)
    dp_group = core.get_group(MeshAxis.DP)
    if cp_group is not None:
        dist.all_reduce(runtime_loss, op=dist.ReduceOp.SUM, group=cp_group)
    if dp_group is not None:
        dist.all_reduce(runtime_loss, op=dist.ReduceOp.AVG, group=dp_group)

    # Flush async grad sync before comparing.
    core._run_phase(RuntimePhase.PRE_STEP)

    reference_optimizer.step()
    core.step_optimizer()
    if isinstance(zero_plugin, Zero3Plugin):
        zero_plugin.materialize_model()

    ep_group = core.get_group(MeshAxis.EP)
    ref_params = _baseline_named_params(reference_model)
    runtime_params = _runtime_named_params(core.model, ep_group)
    step_name, step_diff = _max_diff(ref_params, runtime_params)

    if isinstance(zero_plugin, Zero3Plugin):
        zero_plugin.reshard_model()

    loss_diff_tensor = torch.tensor(abs(reference_loss.item() - runtime_loss.item()), dtype=torch.float64)
    step_diff_tensor = torch.tensor(step_diff, dtype=torch.float64)
    dist.all_reduce(loss_diff_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(step_diff_tensor, op=dist.ReduceOp.MAX)

    if rank == 0:
        reuse_label = f"reuse_tp={args.reuse_tp_for_ep} reuse_cp={args.reuse_cp_for_ep}"
        print(f"Case          : {args.case}  ({reuse_label})")
        print(f"Reference loss: {reference_loss.item():.6f}")
        print(f"Runtime loss  : {runtime_loss.item():.6f}")
        print(f"Loss diff     : {loss_diff_tensor.item():.2e}  (atol={_LOSS_ATOL:.2e})")
        print(f"Step diff     : {step_diff_tensor.item():.2e}  ({step_name}, atol={_STEP_ATOL:.2e})")
        if loss_diff_tensor.item() > _LOSS_ATOL:
            raise AssertionError(f"EP+CP+ZeRO loss equivalence failed: diff={loss_diff_tensor.item():.2e}")
        if step_diff_tensor.item() > _STEP_ATOL:
            raise AssertionError(f"EP+CP+ZeRO step equivalence failed: param={step_name}, diff={step_diff_tensor.item():.2e}")
        print("PASS")

    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    assert args.world_size == args.dp_size * args.cp_size
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
