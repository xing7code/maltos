"""Equivalence test: baseline TinyTransformer vs RuntimeCore TP/SP v2.

This is the migration gate for TP and SP on the RuntimeCore/RuntimePlugin path.

Usage:
  PYTHONPATH=. .venv/bin/python train_system/tests/tiny_transformer_tp_runtime_core_equivalence.py \
    --world-size 2 \
    --tp-size 2

  PYTHONPATH=. .venv/bin/python train_system/tests/tiny_transformer_tp_runtime_core_equivalence.py \
    --world-size 2 \
    --tp-size 2 \
    --use-sp true
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train_system.models import TinyTransformer, TinyTransformerTp, TinyTransformerTpSp
from train_system.parallel.specs import TpSpShardAxis
from train_system.parallel import ParallelPlan
from train_system.runtime import MeshConfig, RuntimeCore
from train_system.runtime.plugins.sp import SequenceParallelPlugin
from train_system.runtime.plugins.tp import TensorParallelPlugin


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

_ATOL = 1e-3
_GRAD_ATOL = 2e-2
_STEP_ATOL = 5e-4
_LR = 1e-2


def _str_to_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in ("1", "true", "t", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--use-sp", type=_str_to_bool, default=False)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29515)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _build_reference(seed: int, batch_size: int, seq_len: int) -> tuple[TinyTransformer, torch.Tensor]:
    torch.manual_seed(seed)
    tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len))
    model = TinyTransformer(**_MODEL_KWARGS)
    return model, tokens


def _baseline_loss(model: TinyTransformer, tokens: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        loss = model((tokens, tokens.clone()))
    return loss.item()


def _rule_by_param_name(model: TinyTransformer) -> dict[str, str]:
    if not hasattr(model, "parallelize_spec"):
        return {}
    rules = {}
    for rule in model.parallelize_spec().rules:
        if rule.shard_axis in (TpSpShardAxis.PARAM_OUT, TpSpShardAxis.PARAM_IN):
            rules[f"{rule.module_path}.weight"] = rule.shard_axis
            rules[f"{rule.module_path}.bias"] = rule.shard_axis
    return rules


def _all_gather_tensor(tensor: torch.Tensor) -> list[torch.Tensor]:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return gathered


def _logical_tensor(name: str, tensor: torch.Tensor, shard_rules: dict[str, str]) -> torch.Tensor:
    shard_axis = shard_rules.get(name)
    if shard_axis == TpSpShardAxis.PARAM_OUT:
        return torch.cat(_all_gather_tensor(tensor), dim=0)
    if shard_axis == TpSpShardAxis.PARAM_IN:
        return torch.cat(_all_gather_tensor(tensor), dim=1)
    return tensor.detach().clone()


def _logical_named_tensors(model: torch.nn.Module, shard_rules: dict[str, str]) -> dict[str, torch.Tensor]:
    return {
        name: _logical_tensor(name, param.detach(), shard_rules)
        for name, param in model.named_parameters()
    }


def _logical_named_grads(model: torch.nn.Module, shard_rules: dict[str, str]) -> dict[str, torch.Tensor]:
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grads[name] = _logical_tensor(name, param.grad.detach(), shard_rules)
    return grads


def _max_diff(lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]) -> tuple[str, float]:
    worst_name = ""
    worst_diff = 0.0
    for name, lhs_tensor in lhs.items():
        rhs_tensor = rhs[name]
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

    baseline_model, tokens = _build_reference(args.seed, args.batch_size, args.seq_len)
    baseline_loss = _baseline_loss(baseline_model, tokens)

    sharded_cls = TinyTransformerTpSp if args.use_sp else TinyTransformerTp
    sharded_model = sharded_cls(**_MODEL_KWARGS)
    sharded_model.load_state_dict(baseline_model.state_dict())

    plugins = [TensorParallelPlugin()]
    if args.use_sp:
        plugins.append(SequenceParallelPlugin())

    core = RuntimeCore(
        mesh=MeshConfig(dp=1, tp=args.tp_size, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=sharded_model,
        optimizer=torch.optim.SGD(sharded_model.parameters(), lr=0.0),
        plugins=plugins,
    )
    core.setup()
    baseline_model.train()
    core.model.train()

    baseline_optimizer = torch.optim.SGD(baseline_model.parameters(), lr=_LR)
    sharded_optimizer = torch.optim.SGD(core.model.parameters(), lr=_LR)

    baseline_optimizer.zero_grad(set_to_none=True)
    sharded_optimizer.zero_grad(set_to_none=True)

    baseline_train_loss = baseline_model((tokens, tokens.clone()))
    sharded_train_loss = core.model((tokens, tokens.clone()))
    baseline_train_loss.backward()
    sharded_train_loss.backward()

    shard_rules = _rule_by_param_name(sharded_model)
    baseline_grads = _logical_named_grads(baseline_model, {})
    sharded_grads = _logical_named_grads(core.model, shard_rules)
    grad_name, grad_diff = _max_diff(baseline_grads, sharded_grads)

    baseline_optimizer.step()
    sharded_optimizer.step()

    baseline_params = _logical_named_tensors(baseline_model, {})
    sharded_params = _logical_named_tensors(core.model, shard_rules)
    step_name, step_diff = _max_diff(baseline_params, sharded_params)

    if rank == 0:
        sharded_loss = sharded_train_loss.item()
        diff = abs(baseline_loss - sharded_loss)
        runtime_label = "RuntimeCore TP+SP" if args.use_sp else "RuntimeCore TP"
        print(f"Baseline loss : {baseline_loss:.6f}")
        print(f"{runtime_label}: {sharded_loss:.6f}")
        print(f"Diff          : {diff:.2e}  (atol={_ATOL:.2e})")
        print(f"Grad diff     : {grad_diff:.2e}  ({grad_name}, atol={_GRAD_ATOL:.2e})")
        print(f"Step diff     : {step_diff:.2e}  ({step_name}, atol={_STEP_ATOL:.2e})")
        if diff > _ATOL:
            raise AssertionError(
                f"RuntimeCore TP equivalence failed: baseline_loss={baseline_loss:.6f}, "
                f"sharded_loss={sharded_loss:.6f}, diff={diff:.2e}, atol={_ATOL:.2e}"
            )
        if grad_diff > _GRAD_ATOL:
            raise AssertionError(
                f"RuntimeCore gradient equivalence failed: param={grad_name}, "
                f"diff={grad_diff:.2e}, atol={_GRAD_ATOL:.2e}"
            )
        if step_diff > _STEP_ATOL:
            raise AssertionError(
                f"RuntimeCore one-step equivalence failed: param={step_name}, "
                f"diff={step_diff:.2e}, atol={_STEP_ATOL:.2e}"
            )
        print("PASS")

    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    assert args.tp_size == args.world_size

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
