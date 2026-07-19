"""Smoke tests for checkpoint runtime_spec manifest recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from state import load_logical_checkpoint, save_runtime_spec, save_sharded_checkpoint
from train.cli import _build_model, _build_runtime
from train.flags import (
    TRAIN_CLI_RUNTIME_SPEC_FORMAT,
    build_runtime_spec,
    parse_args_from,
    parse_runtime_spec_args,
)


def test_manifest_includes_runtime_spec_and_convert_works_without_config() -> None:
    with tempfile.TemporaryDirectory(prefix="checkpoint_runtime_spec_") as tmp:
        root = Path(tmp)
        checkpoint_root = root / "runtime_ckpt"
        checkpoint_dir = checkpoint_root / "step_00000001"
        logical_dir = root / "logical_ckpt"
        args = parse_args_from(
            [
                "--model",
                "tiny",
                "--dim",
                "64",
                "--n-heads",
                "4",
                "--n-kv-heads",
                "4",
                "--hidden-size",
                "128",
                "--n-layers",
                "1",
                "--vocab-size",
                "128",
                "--seq-len",
                "16",
                "--max-steps",
                "1",
                "--micro-batch-size",
                "1",
            ],
            require_data=False,
        )
        model = _build_model(args)
        runtime = _build_runtime(args, model, torch.device("cpu"))
        runtime.setup()
        try:
            save_runtime_spec(checkpoint_root, build_runtime_spec(args))
            save_sharded_checkpoint(runtime.state_manager, checkpoint_dir)
        finally:
            runtime.close()

        runtime_spec = json.loads((checkpoint_root / "runtime_spec.json").read_text(encoding="utf-8"))
        assert isinstance(runtime_spec, dict)
        assert runtime_spec["format"] == TRAIN_CLI_RUNTIME_SPEC_FORMAT
        assert runtime_spec["model"]["model"] == "tiny"
        assert runtime_spec["model"]["seq_len"] == 16
        assert runtime_spec["runtime"]["tp_size"] == 1
        assert "max_steps" not in runtime_spec["model"]
        assert "max_steps" not in runtime_spec["runtime"]
        incomplete_spec = json.loads(json.dumps(runtime_spec))
        incomplete_spec["model"].pop("eps")
        try:
            parse_runtime_spec_args(incomplete_spec)
        except ValueError as exc:
            assert "missing=['eps']" in str(exc)
        else:
            raise AssertionError("incomplete runtime spec must not fall back to current CLI defaults")
        step_manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
        assert "runtime_spec" not in step_manifest

        subprocess.run(
            [sys.executable, "tools/convert_checkpoint.py", "runtime-to-logical", "--checkpoint", checkpoint_dir, "--output", logical_dir],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONPATH": "."},
            check=True,
        )

        logical_tensors = load_logical_checkpoint(logical_dir)
        assert "embed.weight" in logical_tensors
        assert "rope.cos" in logical_tensors
        assert "rope.sin" in logical_tensors


def main() -> None:
    test_manifest_includes_runtime_spec_and_convert_works_without_config()
    print("checkpoint runtime spec smoke ok")


if __name__ == "__main__":
    main()
