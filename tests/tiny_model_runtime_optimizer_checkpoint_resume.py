"""Resume test for RuntimeCore-owned optimizer checkpointing."""

from __future__ import annotations

import argparse
import tempfile

import torch

from models import TinyModel
from runtime import RuntimeCore
from state import load_sharded_checkpoint, save_sharded_checkpoint


_ATOL = 1e-6
_LR = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def _build_model(seed: int, hidden_size: int) -> TinyModel:
    torch.manual_seed(seed)
    return TinyModel(hidden_size=hidden_size)


def _build_core(model: TinyModel) -> RuntimeCore:
    core = RuntimeCore(
        model=model,
        optimizer_factory=lambda params: torch.optim.AdamW(params, lr=_LR, weight_decay=0.0),
        scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
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
    checkpoint_dir = args.checkpoint_dir or tempfile.mkdtemp(prefix="runtime_optim_ckpt_")

    torch.manual_seed(args.seed)
    first_batch = torch.randn(args.batch_size, args.hidden_size)

    continuous_model = _build_model(args.seed, args.hidden_size)
    continuous_core = _build_core(continuous_model)
    continuous_core.run_train_step(first_batch)
    save_sharded_checkpoint(continuous_core.state_manager, checkpoint_dir)
    second_batch = torch.randn(args.batch_size, args.hidden_size)
    continuous_core.run_train_step(second_batch)

    restored_model = _build_model(args.seed, args.hidden_size)
    restored_core = _build_core(restored_model)
    load_sharded_checkpoint(restored_core.state_manager, checkpoint_dir)
    restored_second_batch = torch.randn(args.batch_size, args.hidden_size)
    restored_core.run_train_step(restored_second_batch)

    param_name, param_diff = _max_param_diff(continuous_core.model, restored_core.model)
    assert continuous_core.scheduler is not None
    assert restored_core.scheduler is not None
    scheduler_match = continuous_core.scheduler.state_dict() == restored_core.scheduler.state_dict()
    lr_diff = abs(continuous_core.optimizer.param_groups[0]["lr"] - restored_core.optimizer.param_groups[0]["lr"])
    batch_diff = (second_batch - restored_second_batch).abs().max().item()
    print(f"Checkpoint dir    : {checkpoint_dir}")
    print(f"Resume diff       : {param_diff:.2e}  ({param_name}, atol={_ATOL:.2e})")
    print(f"RNG batch diff    : {batch_diff:.2e}")
    print(f"LR diff           : {lr_diff:.2e}")
    if param_diff > _ATOL:
        raise AssertionError(f"Runtime optimizer checkpoint resume failed: param={param_name}, diff={param_diff:.2e}")
    if not scheduler_match or lr_diff > 0.0:
        raise AssertionError("Runtime scheduler checkpoint resume failed")
    if batch_diff > 0.0:
        raise AssertionError("Runtime RNG checkpoint resume failed")
    if restored_core.state.step != 2:
        raise AssertionError(f"Runtime step restore failed: step={restored_core.state.step}")
    print("PASS")


if __name__ == "__main__":
    main()
