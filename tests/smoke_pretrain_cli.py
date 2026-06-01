"""Smoke tests for pretrain CLI config plumbing."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from tools.pretrain import _build_optimizer_factory, _load_config_defaults


def _optimizer_args(*, fused_adamw: bool) -> argparse.Namespace:
    return argparse.Namespace(
        lr=1e-3,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        fused_adamw=fused_adamw,
    )


def test_fused_adamw_config_alias() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "recipe.yaml"
        path.write_text(
            """
training:
  fused_adamw: true
""".lstrip(),
            encoding="utf-8",
        )
        defaults = _load_config_defaults(str(path))

    assert defaults["fused_adamw"] is True


def test_adamw_factory_enables_fused_backend() -> None:
    param = torch.nn.Parameter(torch.ones(2))
    optimizer = _build_optimizer_factory(_optimizer_args(fused_adamw=True))([param])

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["fused"] is True


def test_adamw_factory_keeps_default_backend() -> None:
    param = torch.nn.Parameter(torch.ones(2))
    optimizer = _build_optimizer_factory(_optimizer_args(fused_adamw=False))([param])

    assert isinstance(optimizer, torch.optim.AdamW)
    if "fused" in optimizer.defaults:
        assert optimizer.defaults["fused"] is None


def main() -> None:
    test_fused_adamw_config_alias()
    test_adamw_factory_enables_fused_backend()
    test_adamw_factory_keeps_default_backend()
    print("pretrain cli smoke ok")


if __name__ == "__main__":
    main()
