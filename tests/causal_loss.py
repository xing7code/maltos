from __future__ import annotations

import torch

from utils.constants import IGNORE_INDEX
from utils.losses import causal_cross_entropy


def test_causal_cross_entropy_matches_pytorch_for_targets() -> None:
    torch.manual_seed(123)
    logits = torch.randn(2, 3, 5, requires_grad=True)
    labels = torch.tensor([[1, 2, IGNORE_INDEX], [4, 0, 3]])
    actual = causal_cross_entropy(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits.view(-1, 5), labels.view(-1), ignore_index=IGNORE_INDEX)
    torch.testing.assert_close(actual, expected)


def test_causal_cross_entropy_is_zero_for_no_targets() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)
    labels = torch.full((2, 3), IGNORE_INDEX)
    loss = causal_cross_entropy(logits, labels)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


if __name__ == "__main__":
    test_causal_cross_entropy_matches_pytorch_for_targets()
    test_causal_cross_entropy_is_zero_for_no_targets()
    print("PASS")
