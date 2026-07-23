from __future__ import annotations

import torch

from models.olmo import OlmoRotaryEmbedding, _apply_olmo_rotary


def test_olmo_rope_uses_rotate_half_pairing() -> None:
    rotary = OlmoRotaryEmbedding(head_dim=4, max_position_embeddings=4, base=10_000.0)
    cos, sin = rotary(1, 2)
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

    actual = _apply_olmo_rotary(x, cos, sin)
    expected = x * cos + torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]) * sin

    torch.testing.assert_close(actual, expected)


def test_olmo_rope_duplicates_frequency_halves() -> None:
    rotary = OlmoRotaryEmbedding(head_dim=8, max_position_embeddings=4, base=10_000.0)
    cos, sin = rotary(1, 2)

    torch.testing.assert_close(cos[..., :4], cos[..., 4:])
    torch.testing.assert_close(sin[..., :4], sin[..., 4:])


if __name__ == "__main__":
    test_olmo_rope_uses_rotate_half_pairing()
    test_olmo_rope_duplicates_frequency_halves()
    print("PASS")
