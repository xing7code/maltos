"""Checkpoint resume test for dataloader cursor state."""

from __future__ import annotations

import argparse
import tempfile

import torch

from data import SimpleTensorDataLoader
from models import TinyModel
from runtime import RuntimeCore
from state import load_sharded_checkpoint, save_sharded_checkpoint


_ATOL = 1e-6
_LR = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def _build_core(seed: int, hidden_size: int) -> RuntimeCore:
    torch.manual_seed(seed)
    model = TinyModel(hidden_size=hidden_size)
    core = RuntimeCore(
        model=model,
        optimizer_factory=lambda params: torch.optim.AdamW(params, lr=_LR, weight_decay=0.0),
    )
    core.setup()
    return core


def _max_param_diff(lhs: TinyModel, rhs: TinyModel) -> tuple[str, float]:
    worst_name = ""
    worst_diff = 0.0
    for (lhs_name, lhs_param), (rhs_name, rhs_param) in zip(lhs.named_parameters(), rhs.named_parameters()):
        assert lhs_name == rhs_name
        diff = (lhs_param.detach() - rhs_param.detach()).abs().max().item()
        if diff > worst_diff:
            worst_name = lhs_name
            worst_diff = diff
    return worst_name, worst_diff


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir or tempfile.mkdtemp(prefix="simple_loader_resume_")

    torch.manual_seed(args.seed)
    data = torch.randn(args.num_samples, args.hidden_size)

    continuous_loader = SimpleTensorDataLoader(data, batch_size=args.batch_size)
    continuous_core = _build_core(args.seed, args.hidden_size)
    continuous_core.state_manager.bind_dataloader(continuous_loader)
    first_batch = continuous_loader.next_batch()
    _, should_step = continuous_core.run_step(first_batch)
    continuous_core.step_optimizer()
    save_sharded_checkpoint(
        continuous_core.state_manager,
        checkpoint_dir,
    )

    continuous_second_batch = continuous_loader.next_batch()
    _, should_step = continuous_core.run_step(continuous_second_batch)
    continuous_core.step_optimizer()

    restored_loader = SimpleTensorDataLoader(data, batch_size=args.batch_size)
    restored_core = _build_core(args.seed, args.hidden_size)
    restored_core.state_manager.bind_dataloader(restored_loader)
    load_sharded_checkpoint(restored_core.state_manager, checkpoint_dir)
    restored_second_batch = restored_loader.next_batch()
    _, should_step = restored_core.run_step(restored_second_batch)
    restored_core.step_optimizer()

    batch_diff = (continuous_second_batch - restored_second_batch).abs().max().item()
    param_name, param_diff = _max_param_diff(continuous_core.model, restored_core.model)
    consumed_match = continuous_loader.consumed_tokens == restored_loader.consumed_tokens

    print(f"Checkpoint dir    : {checkpoint_dir}")
    print(f"Batch diff        : {batch_diff:.2e}")
    print(f"Resume diff       : {param_diff:.2e}  ({param_name}, atol={_ATOL:.2e})")
    if batch_diff > 0.0:
        raise AssertionError(f"dataloader cursor resume failed: batch_diff={batch_diff:.2e}")
    if param_diff > _ATOL:
        raise AssertionError(f"model resume failed: param={param_name}, diff={param_diff:.2e}")
    if not consumed_match:
        raise AssertionError("consumed token counts diverged after resume")
    print("PASS")


if __name__ == "__main__":
    main()
