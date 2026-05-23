"""Validation tests for checkpoint manifest artifacts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch

from train_system.examples import TinyModel
from train_system.runtime import RuntimeCore
from train_system.state import load_sharded_checkpoint, save_sharded_checkpoint


def _build_core(seed: int = 1234, hidden_size: int = 32, grad_accum_steps: int = 1) -> RuntimeCore:
    torch.manual_seed(seed)
    model = TinyModel(hidden_size=hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    core = RuntimeCore(model=model, optimizer=optimizer, grad_accum_steps=grad_accum_steps)
    core.setup()
    return core


def _save_base_checkpoint() -> tuple[Path, RuntimeCore]:
    checkpoint_dir = Path(tempfile.mkdtemp(prefix="manifest_validation_"))
    core = _build_core()
    batch = torch.randn(8, 32)
    core.run_train_step(batch)
    save_sharded_checkpoint(core, checkpoint_dir)
    return checkpoint_dir, core


def _load_manifest(checkpoint_dir: Path) -> dict[str, Any]:
    with open(checkpoint_dir / "manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(checkpoint_dir: Path, manifest: dict[str, Any]) -> None:
    with open(checkpoint_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _expect_value_error(fn: Callable[[], None], expected_substring: str) -> None:
    try:
        fn()
    except ValueError as exc:
        if expected_substring not in str(exc):
            raise AssertionError(
                f"expected ValueError containing '{expected_substring}', got '{exc}'"
            ) from exc
    else:
        raise AssertionError(f"expected ValueError containing '{expected_substring}'")


def test_missing_artifacts_field_fails() -> None:
    checkpoint_dir, _ = _save_base_checkpoint()
    manifest = _load_manifest(checkpoint_dir)
    manifest.pop("artifacts", None)
    _write_manifest(checkpoint_dir, manifest)

    _expect_value_error(
        lambda: load_sharded_checkpoint(_build_core(), checkpoint_dir),
        "manifest missing artifacts field",
    )


def test_world_size_mismatch_fails() -> None:
    checkpoint_dir, _ = _save_base_checkpoint()
    manifest = _load_manifest(checkpoint_dir)
    manifest["world_size"] = 2
    _write_manifest(checkpoint_dir, manifest)

    _expect_value_error(
        lambda: load_sharded_checkpoint(_build_core(), checkpoint_dir),
        "checkpoint world_size mismatch",
    )


def test_optimizer_source_mapping_mismatch_fails() -> None:
    checkpoint_dir, _ = _save_base_checkpoint()
    runtime = _build_core()
    runtime.optimizer_state_source_rank = lambda _rank: 1  # type: ignore[method-assign]

    _expect_value_error(
        lambda: load_sharded_checkpoint(runtime, checkpoint_dir),
        "optimizer source mapping mismatch",
    )


def test_duplicate_artifact_fails() -> None:
    checkpoint_dir, _ = _save_base_checkpoint()
    manifest = _load_manifest(checkpoint_dir)
    model_artifact = next(artifact for artifact in manifest["artifacts"] if artifact["kind"] == "model")
    manifest["artifacts"].append(dict(model_artifact))
    _write_manifest(checkpoint_dir, manifest)

    _expect_value_error(
        lambda: load_sharded_checkpoint(_build_core(), checkpoint_dir),
        "duplicate artifact",
    )


def test_eval_only_manifest_load_succeeds() -> None:
    checkpoint_dir, saved_core = _save_base_checkpoint()
    manifest = _load_manifest(checkpoint_dir)
    manifest["artifacts"] = [artifact for artifact in manifest["artifacts"] if artifact["kind"] != "optimizer"]
    _write_manifest(checkpoint_dir, manifest)

    restored_core = _build_core()
    load_sharded_checkpoint(restored_core, checkpoint_dir)
    if restored_core.state.step != saved_core.state.step:
        raise AssertionError(
            f"expected restored step={saved_core.state.step}, got {restored_core.state.step}"
        )


def test_microbatch_idx_roundtrip_succeeds() -> None:
    checkpoint_dir = Path(tempfile.mkdtemp(prefix="manifest_validation_microbatch_"))
    core = _build_core(grad_accum_steps=2)
    batch = torch.randn(8, 32)
    core.run_train_step(batch)
    if core.state.microbatch_idx != 1:
        raise AssertionError(f"expected saved microbatch_idx=1, got {core.state.microbatch_idx}")
    save_sharded_checkpoint(core, checkpoint_dir)

    restored_core = _build_core(grad_accum_steps=2)
    load_sharded_checkpoint(restored_core, checkpoint_dir)
    if restored_core.state.step != core.state.step:
        raise AssertionError(f"expected restored step={core.state.step}, got {restored_core.state.step}")
    if restored_core.state.microbatch_idx != core.state.microbatch_idx:
        raise AssertionError(
            f"expected restored microbatch_idx={core.state.microbatch_idx}, got {restored_core.state.microbatch_idx}"
        )


def main() -> None:
    test_missing_artifacts_field_fails()
    test_world_size_mismatch_fails()
    test_optimizer_source_mapping_mismatch_fails()
    test_duplicate_artifact_fails()
    test_eval_only_manifest_load_succeeds()
    test_microbatch_idx_roundtrip_succeeds()
    print("PASS")


if __name__ == "__main__":
    main()
