"""Tests for mmap token pretraining dataloader state and checkpoint resume."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from data import PretrainingDataLoader, TokenShardDataset
from models import TinyTransformer
from runtime import RuntimeCore
from state import load_sharded_checkpoint, save_sharded_checkpoint
from utils.constants import INPUT_IDS_KEY, LABELS_KEY


_ATOL = 1e-6
_LR = 1e-2
_MODEL_KWARGS = dict(
    dim=32,
    n_heads=4,
    n_kv_heads=4,
    hidden_size=64,
    eps=1e-5,
    n_layers=1,
    vocab_size=128,
    max_seq_len=16,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def _build_core(seed: int) -> RuntimeCore:
    torch.manual_seed(seed)
    model = TinyTransformer(**_MODEL_KWARGS)
    core = RuntimeCore(
        model=model,
        optimizer_factory=lambda params: torch.optim.AdamW(params, lr=_LR, weight_decay=0.0),
    )
    core.setup()
    return core


def _batch_tuple(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return batch[INPUT_IDS_KEY], batch[LABELS_KEY]


def _max_param_diff(lhs: TinyTransformer, rhs: TinyTransformer) -> tuple[str, float]:
    worst_name = ""
    worst_diff = 0.0
    for (lhs_name, lhs_param), (rhs_name, rhs_param) in zip(lhs.named_parameters(), rhs.named_parameters()):
        assert lhs_name == rhs_name
        diff = (lhs_param.detach() - rhs_param.detach()).abs().max().item()
        if diff > worst_diff:
            worst_name = lhs_name
            worst_diff = diff
    return worst_name, worst_diff


def _assert_dp_stride(dataset: TokenShardDataset, seq_len: int) -> None:
    rank0 = PretrainingDataLoader(dataset, seq_len=seq_len, micro_batch_size=1, dp_rank=0, dp_world_size=2)
    rank1 = PretrainingDataLoader(dataset, seq_len=seq_len, micro_batch_size=1, dp_rank=1, dp_world_size=2)
    rank0_batch = rank0.next_batch()[INPUT_IDS_KEY][0]
    rank1_batch = rank1.next_batch()[INPUT_IDS_KEY][0]
    expected_rank1 = rank0_batch + (seq_len + 1)
    if not torch.equal(rank1_batch, expected_rank1):
        raise AssertionError("DP token-stream stride produced overlapping or unexpected samples")


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir or tempfile.mkdtemp(prefix="pretrain_loader_resume_"))
    testdata_dir = Path(__file__).parent / "testdata"
    dataset = TokenShardDataset(
        [
            testdata_dir / "tokens_00000.bin",
            testdata_dir / "tokens_00001.bin",
        ]
    )
    _assert_dp_stride(dataset, args.seq_len)

    continuous_loader = PretrainingDataLoader(
        dataset,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        dp_rank=0,
        dp_world_size=1,
        seed=args.seed,
    )
    continuous_core = _build_core(args.seed)
    continuous_core.state_manager.bind_dataloader(continuous_loader)
    first_batch = continuous_loader.next_batch()
    _, should_step = continuous_core.run_step(_batch_tuple(first_batch))
    continuous_core.step_optimizer()
    saved_runtime_step = continuous_core.state.step
    saved_microbatch_idx = continuous_core.state.step_context.microbatch_idx
    saved_loader_state = continuous_loader.state_dict()
    save_sharded_checkpoint(
        continuous_core.state_manager,
        checkpoint_dir,
    )

    continuous_second_batch = continuous_loader.next_batch()
    _, should_step = continuous_core.run_step(_batch_tuple(continuous_second_batch))
    continuous_core.step_optimizer()

    restored_loader = PretrainingDataLoader(
        dataset,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        dp_rank=0,
        dp_world_size=1,
        seed=args.seed,
    )
    restored_core = _build_core(args.seed)
    restored_core.state_manager.bind_dataloader(restored_loader)
    load_sharded_checkpoint(restored_core.state_manager, checkpoint_dir)
    restored_loader_state = restored_loader.state_dict()
    if restored_core.state.step != saved_runtime_step:
        raise AssertionError(f"runtime step restore failed: expected={saved_runtime_step}, got={restored_core.state.step}")
    if restored_core.state.step_context.microbatch_idx != saved_microbatch_idx:
        raise AssertionError(
            "runtime microbatch_idx restore failed: "
            f"expected={saved_microbatch_idx}, got={restored_core.state.step_context.microbatch_idx}"
        )
    if restored_loader_state != saved_loader_state:
        raise AssertionError(f"dataloader state restore failed: expected={saved_loader_state}, got={restored_loader_state}")
    restored_second_batch = restored_loader.next_batch()
    _, should_step = restored_core.run_step(_batch_tuple(restored_second_batch))
    restored_core.step_optimizer()

    batch_diff = (continuous_second_batch[INPUT_IDS_KEY] - restored_second_batch[INPUT_IDS_KEY]).abs().max().item()
    param_name, param_diff = _max_param_diff(continuous_core.model, restored_core.model)
    print(f"Checkpoint dir    : {checkpoint_dir}")
    print(f"Batch diff        : {batch_diff:.2e}")
    print(f"Resume diff       : {param_diff:.2e}  ({param_name}, atol={_ATOL:.2e})")
    if batch_diff > 0.0:
        raise AssertionError(f"pretraining loader resume failed: batch_diff={batch_diff:.2e}")
    if param_diff > _ATOL:
        raise AssertionError(f"model resume failed: param={param_name}, diff={param_diff:.2e}")
    print("PASS")


if __name__ == "__main__":
    main()
