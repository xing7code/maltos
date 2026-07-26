"""CLI/YAML-facing contract for the HDP profiling cost model."""
from __future__ import annotations

from train.cli import _build_hdp_config, _global_batch_tokens
from train.flags import parse_args_from


def main() -> None:
    args = parse_args_from(
        [
            "--hdp-balanced",
            "--hdp-partition-tokens", "16",
            "--hdp-cost-alpha", "1.25",
            "--hdp-cost-beta", "3.5",
            "--hdp-cost-gamma", "7",
            "--hdp-balance-delta", "9",
        ],
        require_data=False,
    )
    config = _build_hdp_config(args)
    assert config.partition_tokens == 16
    assert config.cost_model.alpha == 1.25
    assert config.cost_model.beta == 3.5
    assert config.cost_model.gamma == 7.0
    assert config.balance_delta == 9
    assert _global_batch_tokens(args) == args.micro_batch_size * args.seq_len * args.grad_accum_steps
    args.hdp_balanced = False
    assert _global_batch_tokens(args) == args.micro_batch_size * args.seq_len * args.dp_size * args.grad_accum_steps
    print("HDP CLI profiling cost model PASS")


if __name__ == "__main__":
    main()
