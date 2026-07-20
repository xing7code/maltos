from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import torch
import torch.distributed as dist

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != _THIS_DIR]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from state import (
    iter_logical_tensors_from_runtime_checkpoint,
    load_runtime_spec,
    save_logical_checkpoint,
    save_runtime_spec,
)
from state.checkpoint import save_sharded_checkpoint_from_model_state
from state.logical_checkpoint import LogicalCheckpointTensorReader, _build_runtime_model_state_from_logical_tensors
from train.flags import parse_args_from as parse_training_args_from
from train.flags import build_runtime_spec, parse_runtime_spec_args
from utils.distributed import distributed_barrier


def _load_train_cli_module():
    cli_path = _REPO_ROOT / "train" / "cli.py"
    spec = importlib.util.spec_from_file_location("maltos_train_cli", cli_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load train CLI module from {cli_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TRAIN_CLI = _load_train_cli_module()
_build_model = _TRAIN_CLI._build_model
_build_runtime = _TRAIN_CLI._build_runtime
_maybe_init_distributed = _TRAIN_CLI._maybe_init_distributed
_select_device = _TRAIN_CLI._select_device


def build_arg_parser_for_convert() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert between runtime and logical checkpoints.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    runtime_to_logical = subparsers.add_parser("runtime-to-logical")
    runtime_to_logical.add_argument("--checkpoint", type=str, required=True, help="Source runtime checkpoint dir.")
    runtime_to_logical.add_argument("--output", type=str, required=True, help="Output logical checkpoint dir.")

    logical_to_runtime = subparsers.add_parser("logical-to-runtime")
    logical_to_runtime.add_argument("--config", type=str, required=True, help="Recipe YAML for the target runtime.")
    logical_to_runtime.add_argument("--checkpoint", type=str, required=True, help="Source logical checkpoint dir.")
    logical_to_runtime.add_argument("--output", type=str, required=True, help="Output runtime checkpoint dir.")
    logical_to_runtime.add_argument(
        "recipe_overrides",
        nargs=argparse.REMAINDER,
        help="Optional recipe overrides after '--', reusing train/cli.py flags.",
    )
    return parser


def main() -> None:
    args = build_arg_parser_for_convert().parse_args()
    recipe_args = _resolve_recipe_args(args)
    if args.command == "runtime-to-logical":
        _runtime_to_logical(recipe_args, Path(args.checkpoint), Path(args.output))
        return

    _maybe_init_distributed(recipe_args)
    device = _select_device()
    rank = dist.get_rank() if dist.is_initialized() else 0
    model = _build_model(recipe_args)
    runtime = _build_runtime(recipe_args, model, device, weights_only=True)
    runtime.setup()
    try:
        if args.command == "logical-to-runtime":
            _logical_to_runtime(runtime, Path(args.checkpoint), Path(args.output), recipe_args=recipe_args, rank=rank)
    finally:
        runtime.close()
        if dist.is_initialized():
            distributed_barrier()
            dist.destroy_process_group()


def _parse_recipe_args(config_path: str, overrides: list[str]) -> argparse.Namespace:
    normalized_overrides = list(overrides)
    if normalized_overrides and normalized_overrides[0] == "--":
        normalized_overrides = normalized_overrides[1:]
    return parse_training_args_from(["--config", config_path, *normalized_overrides], require_data=False)


def _resolve_recipe_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.command == "logical-to-runtime":
        return _parse_recipe_args(args.config, args.recipe_overrides)
    return parse_runtime_spec_args(load_runtime_spec(args.checkpoint))


def _runtime_to_logical(recipe_args: argparse.Namespace, checkpoint_dir: Path, output_dir: Path) -> None:
    model = _build_model(recipe_args)
    tensors = dict(
        iter_logical_tensors_from_runtime_checkpoint(
            checkpoint_dir,
            model=model,
            ep_size=int(getattr(recipe_args, "ep_size", 1)),
            dp_size=int(getattr(recipe_args, "dp_size", 1)),
            tp_size=int(getattr(recipe_args, "tp_size", 1)),
            pp_size=int(getattr(recipe_args, "pp_size", 1)),
            cp_size=int(getattr(recipe_args, "cp_size", 1)),
        )
    )
    save_logical_checkpoint(output_dir, tensors)
    print(f"saved logical checkpoint to {output_dir}")


def _logical_to_runtime(runtime, checkpoint_dir: Path, output_dir: Path, *, recipe_args: argparse.Namespace, rank: int) -> None:
    tensors = LogicalCheckpointTensorReader(checkpoint_dir)
    model_state, rank_entries = _build_runtime_model_state_from_logical_tensors(runtime, tensors)
    if rank == 0:
        save_runtime_spec(output_dir, build_runtime_spec(recipe_args))
    if dist.is_initialized():
        distributed_barrier()
    # TODO: support full logical->runtime conversion, including optimizer,
    # scheduler, RNG, dataloader, and in-flight trainer state, so training can
    # resume mid-run under a new runtime layout. For now this path emits a
    # weights-only runtime checkpoint.
    save_sharded_checkpoint_from_model_state(
        runtime.state_manager,
        output_dir / "step_00000000",
        model_state=model_state,
        rank_entries=rank_entries,
        include_training_state=False,
    )
    if rank == 0:
        print(f"saved runtime checkpoint to {output_dir}")
    if dist.is_initialized():
        distributed_barrier()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
