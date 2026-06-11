"""Equivalence test: baseline TinyMoETransformer vs RuntimeCore EP+PP+ZeRO variants.

Topology: dp=2, pp=2, ep=2 → world_size=4.
Each PP stage holds 2 of 4 transformer layers; EP splits experts across 2 DP ranks
sharing the same PP stage.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distributed_test_utils import moe_named_tensors as _moe_named_tensors
from helpers import causal_lm_batch
from models import TinyMoETransformer
from models.tiny_transformer import RmsNorm
from parallel import ParallelPlan
from parallel.schedule import PipelineScheduleConfig
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.plugins.ddp import BucketDataParallelPlugin
from runtime.plugins.ep import ExpertParallelPlugin
from runtime.plugins.pp import PipelineParallelPlugin
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
    num_experts=4,
)

_LOSS_ATOL = 1e-3
_STEP_ATOL = 5e-4
_LR = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("ep_pp_zero0", "ep_pp_zero1", "ep_pp_zero2", "ep_pp_zero3"),
        default="ep_pp_zero1",
    )
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--pp-microbatches", type=int, default=2)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29640)
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

def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )

    mesh = MeshConfig(dp=args.dp_size, tp=1, pp=args.pp_size, cp=1, ep=args.ep_size)
    dp_idx, pp_idx, cp_idx, tp_idx = mesh.rank_coordinates(rank)
    local_batch_size = args.batch_size

    # Reference: single process, full batch, full model.
    baseline_model, global_tokens = _build_reference(args.seed, args.dp_size, args.batch_size, args.seq_len)
    baseline_model.train()
    baseline_optimizer = torch.optim.SGD(baseline_model.parameters(), lr=_LR)
    baseline_optimizer.zero_grad(set_to_none=True)
    baseline_loss_val = baseline_model(causal_lm_batch(global_tokens))
    baseline_loss_val.backward()

    # Sharded: PP + EP + grad plugin — copy weights BEFORE optimizer step so both
    # start from the same θ₀ and we can compare losses directly.
    sharded_model = TinyMoETransformer(**_MODEL_KWARGS)
    sharded_model.load_state_dict(baseline_model.state_dict())
    sharded_model.train()

    baseline_optimizer.step()
    baseline_params = {name: param.detach().clone().cpu() for name, param in baseline_model.named_parameters()}

    zero_stage_map = {"ep_pp_zero0": 0, "ep_pp_zero1": 1, "ep_pp_zero2": 2, "ep_pp_zero3": 3}
    zero_stage = zero_stage_map[args.case]
    zero3: Zero3Plugin | None = None
    if zero_stage == 0:
        grad_plugin: BucketDataParallelPlugin | Zero1Plugin | Zero2Plugin | Zero3Plugin = BucketDataParallelPlugin(bucket_mb_size=0)
    elif zero_stage == 1:
        grad_plugin = Zero1Plugin(bucket_mb_size=0)
    elif zero_stage == 2:
        grad_plugin = Zero2Plugin(bucket_mb_size=0)
    else:
        zero3 = Zero3Plugin(wrap_cls={torch.nn.Linear, torch.nn.Embedding, RmsNorm})
        grad_plugin = zero3

    plugins = [
        PipelineParallelPlugin(schedule="1f1b"),
        ExpertParallelPlugin(),
        grad_plugin,
    ]
    core = RuntimeCore(
        mesh=mesh,
        plan=ParallelPlan(
            zero_stage=zero_stage,
            pp_schedule=PipelineScheduleConfig(microbatches=args.pp_microbatches),
        ),
        model=sharded_model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=plugins,
    )
    core.setup()
    core.model.train()

    local_tokens = _tokens_for_dp(args.seed, dp_idx, local_batch_size, args.seq_len)
    runtime_loss, should_step = core.run_step(causal_lm_batch(local_tokens))
    if not should_step:
        raise AssertionError("EP+PP test expected should_step=True")

    core.step_optimizer()

    if zero3 is not None:
        zero3.materialize_model()

    # Gather params owned by this PP stage (including EP-gathered expert params).
    ep_group = core.get_group(MeshAxis.EP)
    runtime_params = _moe_named_tensors(core.model, {}, None, ep_group, device=torch.device("cpu"))

    # Compare only params owned by this PP stage.
    local_param_diff = 0.0
    local_param_name = ""
    for name, param in runtime_params.items():
        if name not in baseline_params:
            continue
        ref = baseline_params[name].to(param.device, param.dtype)
        diff = float((param - ref).abs().max().item())
        if diff > local_param_diff:
            local_param_diff = diff
            local_param_name = name

    if zero3 is not None:
        zero3.reshard_model()

    # Average runtime loss over DP to compare with full-batch baseline.
    avg_loss = runtime_loss.detach().clone()
    dp_group = core.get_group(MeshAxis.DP)
    if dp_group is not None:
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG, group=dp_group)

    loss_diff_tensor = torch.tensor(abs(avg_loss.item() - baseline_loss_val.item()), dtype=torch.float64)
    param_diff_tensor = torch.tensor(local_param_diff, dtype=torch.float64)
    dist.all_reduce(loss_diff_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(param_diff_tensor, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(f"Case          : {args.case}")
        print(f"Baseline loss : {baseline_loss_val.item():.6f}")
        print(f"Runtime loss  : {avg_loss.item():.6f}")
        print(f"Loss diff     : {loss_diff_tensor.item():.2e}  (atol={_LOSS_ATOL:.2e})")
        print(f"Step diff     : {param_diff_tensor.item():.2e}  ({local_param_name}, atol={_STEP_ATOL:.2e})")
        if loss_diff_tensor.item() > _LOSS_ATOL:
            raise AssertionError(f"EP+PP loss equivalence failed: diff={loss_diff_tensor.item():.2e}")
        if param_diff_tensor.item() > _STEP_ATOL:
            raise AssertionError(
                f"EP+PP step equivalence failed: param={local_param_name}, diff={param_diff_tensor.item():.2e}"
            )
        print("PASS")

    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    assert args.world_size == args.dp_size * args.pp_size, "world_size must equal dp_size * pp_size"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
