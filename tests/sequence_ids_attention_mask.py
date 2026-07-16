from __future__ import annotations

import torch

from models.llama import _LlamaAttentionCore
from models.tiny_transformer import _LocalCausalAttentionCore
from runtime.layers.attention import _eager_cp_causal_attention


def _example_values() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.ones((1, 1, 6, 1), dtype=torch.float32)
    k = torch.ones((1, 1, 6, 1), dtype=torch.float32)
    v = torch.tensor([10.0, 20.0, 30.0, 100.0, 200.0, 300.0], dtype=torch.float32).view(1, 1, 6, 1)
    return q, k, v


def test_local_attention_masks_cross_example_tokens() -> None:
    q, k, v = _example_values()
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long)
    sequence_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)

    out = _LocalCausalAttentionCore()(q, k, v, 0, position_ids, sequence_ids)

    expected = torch.tensor([10.0, 15.0, 20.0, 100.0, 150.0, 200.0], dtype=torch.float32).view(1, 1, 6, 1)
    assert torch.allclose(out, expected, atol=1e-5)


def test_llama_attention_core_masks_cross_example_tokens() -> None:
    q, k, v = _example_values()
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long)
    sequence_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)

    out = _LlamaAttentionCore("eager")(q, k, v, 0, position_ids, sequence_ids)

    expected = torch.tensor([10.0, 15.0, 20.0, 100.0, 150.0, 200.0], dtype=torch.float32).view(1, 1, 6, 1)
    assert torch.allclose(out, expected, atol=1e-5)


def test_cp_attention_masks_cross_example_tokens() -> None:
    q = torch.ones((1, 1, 3, 1), dtype=torch.float32)
    k = torch.ones((1, 1, 6, 1), dtype=torch.float32)
    v = torch.tensor([10.0, 20.0, 30.0, 100.0, 200.0, 300.0], dtype=torch.float32).view(1, 1, 6, 1)
    q_positions = torch.tensor([[0, 1, 2]], dtype=torch.long)
    k_positions = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long)
    q_sequence_ids = torch.tensor([[1, 1, 1]], dtype=torch.long)
    k_sequence_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)

    out = _eager_cp_causal_attention(
        q,
        k,
        v,
        q_positions=q_positions,
        k_positions=k_positions,
        q_sequence_ids=q_sequence_ids,
        k_sequence_ids=k_sequence_ids,
    )

    expected = torch.tensor([100.0, 150.0, 200.0], dtype=torch.float32).view(1, 1, 3, 1)
    assert torch.allclose(out, expected, atol=1e-5)


def main() -> None:
    test_local_attention_masks_cross_example_tokens()
    test_llama_attention_core_masks_cross_example_tokens()
    test_cp_attention_masks_cross_example_tokens()
    print("sequence_ids attention mask ok")


if __name__ == "__main__":
    main()
