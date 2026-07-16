from __future__ import annotations

import torch

from models.llama import _LlamaAttentionCore
from models.tiny_transformer import _LocalCausalAttentionCore
from runtime.layers.attention import _eager_cp_causal_attention
from utils.attention_backend import AttentionBackend
from utils.constants import PAD_SEQUENCE_ID
from utils.flash_attention import pack_varlen_prefix_qkv, pack_varlen_qkv


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

    out = _LlamaAttentionCore(AttentionBackend.EAGER)(q, k, v, 0, position_ids, sequence_ids)

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


def test_pack_varlen_qkv_builds_expected_cu_seqlens() -> None:
    q = torch.arange(2 * 2 * 6 * 3, dtype=torch.float32).view(2, 2, 6, 3)
    sequence_ids = torch.tensor(
        [
            [0, 0, 0, 1, 1, PAD_SEQUENCE_ID],
            [2, 2, 3, 3, PAD_SEQUENCE_ID, PAD_SEQUENCE_ID],
        ],
        dtype=torch.long,
    )

    packed = pack_varlen_qkv(q, q, q, sequence_ids)

    assert packed.q.shape == (9, 2, 3)
    assert packed.k.shape == (9, 2, 3)
    assert packed.v.shape == (9, 2, 3)
    assert packed.cu_seqlens.tolist() == [0, 3, 5, 7, 9]
    assert packed.max_seqlen == 3
    assert packed.token_mask.tolist() == [
        [True, True, True, True, True, False],
        [True, True, True, True, False, False],
    ]


def test_pack_varlen_prefix_qkv_builds_expected_qk_segments() -> None:
    q = torch.arange(1 * 2 * 5 * 3, dtype=torch.float32).view(1, 2, 5, 3)
    k = torch.arange(1 * 2 * 10 * 3, dtype=torch.float32).view(1, 2, 10, 3)
    q_positions = torch.tensor([[5, 6, 0, 1, 2]], dtype=torch.long)
    k_positions = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0, 1, 2]], dtype=torch.long)
    q_sequence_ids = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long)
    k_sequence_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], dtype=torch.long)

    packed = pack_varlen_prefix_qkv(
        q,
        k,
        k,
        q_positions=q_positions,
        k_positions=k_positions,
        q_sequence_ids=q_sequence_ids,
        k_sequence_ids=k_sequence_ids,
    )

    assert packed.q.shape == (5, 2, 3)
    assert packed.k.shape == (10, 2, 3)
    assert packed.v.shape == (10, 2, 3)
    assert packed.cu_seqlens_q.tolist() == [0, 2, 5]
    assert packed.cu_seqlens_k.tolist() == [0, 7, 10]
    assert packed.max_seqlen_q == 3
    assert packed.max_seqlen_k == 7
    assert packed.token_mask.tolist() == [[True, True, True, True, True]]


def main() -> None:
    test_local_attention_masks_cross_example_tokens()
    test_llama_attention_core_masks_cross_example_tokens()
    test_cp_attention_masks_cross_example_tokens()
    test_pack_varlen_qkv_builds_expected_cu_seqlens()
    test_pack_varlen_prefix_qkv_builds_expected_qk_segments()
    print("sequence_ids attention mask ok")


if __name__ == "__main__":
    main()
