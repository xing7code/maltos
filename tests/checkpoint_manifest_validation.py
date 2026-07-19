"""Validation tests for checkpoint manifest artifacts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from models import TinyModel
from runtime import RuntimeCore
from state import load_sharded_checkpoint, save_sharded_checkpoint


class _BufferedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.register_buffer("rope_cos", torch.arange(16, dtype=torch.float32).view(4, 4))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.proj(batch + self.rope_cos).sum()


def _build_core(seed: int = 1234, hidden_size: int = 32, grad_accum_steps: int = 1) -> RuntimeCore:
    torch.manual_seed(seed)
    model = TinyModel(hidden_size=hidden_size)
    core = RuntimeCore(
        model=model,
        grad_accum_steps=grad_accum_steps,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=1e-2),
    )
    core.setup()
    return core


def _build_buffer_core(seed: int = 1234) -> RuntimeCore:
    torch.manual_seed(seed)
    core = RuntimeCore(
        model=_BufferedModel(),
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=1e-2),
    )
    core.setup()
    return core


def _save_base_checkpoint() -> tuple[Path, RuntimeCore]:
    checkpoint_dir = Path(tempfile.mkdtemp(prefix="manifest_validation_"))
    core = _build_core(grad_accum_steps=2)
    batch = torch.randn(8, 32)
    _, should_step = core.run_step(batch)
    core.step_optimizer()
    save_sharded_checkpoint(core.state_manager, checkpoint_dir)
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
        lambda: load_sharded_checkpoint(_build_core().state_manager, checkpoint_dir),
        "manifest missing artifacts field",
    )


def test_world_size_mismatch_fails() -> None:
    checkpoint_dir, _ = _save_base_checkpoint()
    manifest = _load_manifest(checkpoint_dir)
    manifest["world_size"] = 2
    _write_manifest(checkpoint_dir, manifest)

    _expect_value_error(
        lambda: load_sharded_checkpoint(_build_core().state_manager, checkpoint_dir),
        "checkpoint world_size mismatch",
    )


def test_optimizer_source_mapping_mismatch_fails() -> None:
    checkpoint_dir, _ = _save_base_checkpoint()
    runtime = _build_core()
    runtime.optimizer_checkpoint_rank = lambda _rank: 1  # type: ignore[method-assign]

    _expect_value_error(
        lambda: load_sharded_checkpoint(runtime.state_manager, checkpoint_dir),
        "optimizer source mapping mismatch",
    )


def test_duplicate_artifact_fails() -> None:
    checkpoint_dir, _ = _save_base_checkpoint()
    manifest = _load_manifest(checkpoint_dir)
    model_artifact = next(artifact for artifact in manifest["artifacts"] if artifact["kind"] == "model")
    manifest["artifacts"].append(dict(model_artifact))
    _write_manifest(checkpoint_dir, manifest)

    _expect_value_error(
        lambda: load_sharded_checkpoint(_build_core().state_manager, checkpoint_dir),
        "duplicate artifact",
    )


def test_missing_model_source_rank_fails() -> None:
    checkpoint_dir, _ = _save_base_checkpoint()
    manifest = _load_manifest(checkpoint_dir)
    manifest["ranks"][0]["entries"][0]["source_rank"] = None
    _write_manifest(checkpoint_dir, manifest)

    _expect_value_error(
        lambda: load_sharded_checkpoint(_build_core().state_manager, checkpoint_dir),
        "manifest missing model source rank",
    )


def test_model_source_rank_out_of_range_fails() -> None:
    checkpoint_dir, _ = _save_base_checkpoint()
    manifest = _load_manifest(checkpoint_dir)
    manifest["ranks"][0]["entries"][0]["source_rank"] = manifest["world_size"]
    _write_manifest(checkpoint_dir, manifest)

    _expect_value_error(
        lambda: load_sharded_checkpoint(_build_core().state_manager, checkpoint_dir),
        "manifest model source rank out of range",
    )


def test_missing_model_entry_in_source_artifact_fails() -> None:
    checkpoint_dir, _ = _save_base_checkpoint()
    manifest = _load_manifest(checkpoint_dir)
    manifest["ranks"][0]["entries"][0]["state_key"] = "__missing_entry__"
    _write_manifest(checkpoint_dir, manifest)

    _expect_value_error(
        lambda: load_sharded_checkpoint(_build_core().state_manager, checkpoint_dir),
        "model state missing entry=__missing_entry__",
    )


def test_eval_only_manifest_load_succeeds() -> None:
    checkpoint_dir, saved_core = _save_base_checkpoint()
    manifest = _load_manifest(checkpoint_dir)
    manifest["artifacts"] = [artifact for artifact in manifest["artifacts"] if artifact["kind"] != "optimizer"]
    _write_manifest(checkpoint_dir, manifest)

    restored_core = _build_core(grad_accum_steps=2)
    load_sharded_checkpoint(restored_core.state_manager, checkpoint_dir)
    if restored_core.state.step != saved_core.state.step:
        raise AssertionError(
            f"expected restored step={saved_core.state.step}, got {restored_core.state.step}"
        )


def test_microbatch_idx_roundtrip_succeeds() -> None:
    checkpoint_dir = Path(tempfile.mkdtemp(prefix="manifest_validation_microbatch_"))
    core = _build_core(grad_accum_steps=2)
    batch = torch.randn(8, 32)
    _, should_step = core.run_step(batch)
    if core.state.step_context.microbatch_idx != 1:
        raise AssertionError(f"expected saved microbatch_idx=1, got {core.state.step_context.microbatch_idx}")
    save_sharded_checkpoint(core.state_manager, checkpoint_dir)

    restored_core = _build_core(grad_accum_steps=2)
    load_sharded_checkpoint(restored_core.state_manager, checkpoint_dir)
    if restored_core.state.step != core.state.step:
        raise AssertionError(f"expected restored step={core.state.step}, got {restored_core.state.step}")
    if restored_core.state.step_context.microbatch_idx != core.state.step_context.microbatch_idx:
        raise AssertionError(
            f"expected restored microbatch_idx={core.state.step_context.microbatch_idx}, got {restored_core.state.step_context.microbatch_idx}"
        )


def test_persistent_buffer_roundtrip_succeeds() -> None:
    checkpoint_dir = Path(tempfile.mkdtemp(prefix="manifest_validation_buffer_"))
    core = _build_buffer_core()
    core.model.rope_cos.add_(7.0)
    save_sharded_checkpoint(core.state_manager, checkpoint_dir)

    manifest = _load_manifest(checkpoint_dir)
    manifest_entries = manifest["ranks"][0]["entries"]
    if not any(entry["state_key"] == "rope_cos" for entry in manifest_entries):
        raise AssertionError("expected manifest to include persistent buffer entry for rope_cos")

    restored_core = _build_buffer_core(seed=4321)
    restored_core.model.rope_cos.zero_()
    load_sharded_checkpoint(restored_core.state_manager, checkpoint_dir, weights_only=True)
    if not torch.equal(restored_core.model.rope_cos, core.model.rope_cos):
        raise AssertionError("expected persistent buffer to roundtrip through sharded checkpoint")


def main() -> None:
    test_missing_artifacts_field_fails()
    test_world_size_mismatch_fails()
    test_optimizer_source_mapping_mismatch_fails()
    test_duplicate_artifact_fails()
    test_missing_model_source_rank_fails()
    test_model_source_rank_out_of_range_fails()
    test_missing_model_entry_in_source_artifact_fails()
    test_eval_only_manifest_load_succeeds()
    test_microbatch_idx_roundtrip_succeeds()
    test_persistent_buffer_roundtrip_succeeds()
    print("PASS")


if __name__ == "__main__":
    main()
