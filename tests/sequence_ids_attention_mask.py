from __future__ import annotations

import torch

from models.llama import _LlamaAttentionCore
from models.tiny_transformer import _LocalCausalAttentionCore
from runtime.layers import functional as layer_functional
from runtime.layers.flash_utils import pack_dense_varlen_qkv, pack_varlen_prefix_qkv, pack_varlen_qkv
from runtime.layers.flash_utils import FlashAttnBlockOutput
from runtime.layers.functional import _AsyncRingExchange, _flash_ring_backward, _flash_ring_forward
from runtime.layers.ring_layout import expected_zigzag_ring_positions
from utils.attention_backend import AttentionBackend, causal_attention
from utils.constants import PAD_SEQUENCE_ID


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

    out = causal_attention(
        q,
        k,
        v,
        attention_backend=AttentionBackend.EAGER,
        warning_prefix="test",
        q_positions=q_positions,
        k_positions=k_positions,
        q_sequence_ids=q_sequence_ids,
        k_sequence_ids=k_sequence_ids,
        flash_varlen_mode="prefix",
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
    assert packed.q_indices.tolist() == [0, 1, 2, 3, 4]
    assert packed.k_indices.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_pack_varlen_prefix_qkv_can_skip_empty_prefix_runs() -> None:
    q = torch.arange(1 * 2 * 4 * 3, dtype=torch.float32).view(1, 2, 4, 3)
    k = torch.arange(1 * 2 * 4 * 3, dtype=torch.float32).view(1, 2, 4, 3)
    q_positions = torch.tensor([[0, 1, 6, 7]], dtype=torch.long)
    k_positions = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)

    packed = pack_varlen_prefix_qkv(
        q,
        k,
        k,
        q_positions=q_positions,
        k_positions=k_positions,
        allow_empty_kv=True,
    )

    assert packed.q.shape == (2, 2, 3)
    assert packed.k.shape == (4, 2, 3)
    assert packed.v.shape == (4, 2, 3)
    assert packed.q_indices.tolist() == [2, 3]
    assert packed.k_indices.tolist() == [0, 1, 2, 3]
    assert packed.cu_seqlens_q.tolist() == [0, 2]
    assert packed.cu_seqlens_k.tolist() == [0, 4]
    assert packed.max_seqlen_q == 2
    assert packed.max_seqlen_k == 4


def test_expected_zigzag_ring_positions_match_cp_layout() -> None:
    positions = expected_zigzag_ring_positions(
        4,
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
    )
    assert positions.tolist() == [0, 1, 6, 7]


def test_pack_dense_varlen_qkv_builds_uniform_cu_seqlens() -> None:
    q = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).view(2, 3, 4, 5)
    packed = pack_dense_varlen_qkv(q, q, q)

    assert packed.q.shape == (8, 3, 5)
    assert packed.k.shape == (8, 3, 5)
    assert packed.v.shape == (8, 3, 5)
    assert packed.cu_seqlens.tolist() == [0, 4, 8]
    assert packed.max_seqlen == 4


class _FakeGroup:
    def __init__(self, *, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size


def _dense_attention_block(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> FlashAttnBlockOutput:
    scores = torch.matmul(q, k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
    if causal:
        causal_mask = torch.ones((q.size(-2), k.size(-2)), dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal_mask.view(1, 1, q.size(-2), k.size(-2)), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return FlashAttnBlockOutput(
        out=torch.matmul(probs, v),
        lse=torch.logsumexp(scores.float(), dim=-1),
    )


def _global_dense_causal_attention(
    q: torch.Tensor,
    k_by_position: torch.Tensor,
    v_by_position: torch.Tensor,
    *,
    q_positions: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(q)
    for idx, q_position in enumerate(q_positions[0].tolist()):
        visible_k = k_by_position[:, :, : q_position + 1, :]
        visible_v = v_by_position[:, :, : q_position + 1, :]
        block = _dense_attention_block(q[:, :, idx : idx + 1, :], visible_k, visible_v, causal=False)
        out[:, :, idx : idx + 1, :] = block.out
    return out


def _expected_zigzag_calls(*, rank: int, world_size: int, local_seq_len: int) -> list[tuple[int, int, bool]]:
    half_seq_len = local_seq_len // 2
    calls = [(local_seq_len, local_seq_len, True)]
    for step in range(1, world_size):
        if step <= rank:
            calls.append((local_seq_len, half_seq_len, False))
        else:
            calls.append((half_seq_len, local_seq_len, False))
    return calls


def test_flash_ring_forward_uses_dense_zigzag_half_schedule() -> None:
    batch_size = 1
    num_heads = 1
    local_seq_len = 4
    head_dim = 3

    original_get_world_size = layer_functional.dist.get_world_size
    original_get_rank = layer_functional.dist.get_rank
    original_exchange = layer_functional._ring_exchange_tensor_async
    original_dense = layer_functional.flash_attn_dense_with_lse
    try:
        for world_size in (2, 4):
            total_seq_len = local_seq_len * world_size
            all_q = torch.randn(
                (batch_size, num_heads, total_seq_len, head_dim),
                generator=torch.Generator().manual_seed(1 + world_size),
            )
            all_k = torch.randn(
                (batch_size, num_heads, total_seq_len, head_dim),
                generator=torch.Generator().manual_seed(2 + world_size),
            )
            all_v = torch.randn(
                (batch_size, num_heads, total_seq_len, head_dim),
                generator=torch.Generator().manual_seed(3 + world_size),
            )
            positions = [
                expected_zigzag_ring_positions(local_seq_len, rank=rank, world_size=world_size, device=torch.device("cpu")).view(1, -1)
                for rank in range(world_size)
            ]
            q_chunks = [all_q.index_select(2, pos.flatten()) for pos in positions]
            k_chunks = [all_k.index_select(2, pos.flatten()) for pos in positions]
            v_chunks = [all_v.index_select(2, pos.flatten()) for pos in positions]

            for rank in range(world_size):
                calls: list[tuple[int, int, bool]] = []
                group = _FakeGroup(rank=rank, world_size=world_size)

                def fake_get_world_size(fake_group) -> int:
                    return fake_group.world_size

                def fake_get_rank(fake_group) -> int:
                    return fake_group.rank

                def fake_exchange(x, fake_group, send_to, recv_from, *, alloc_key):
                    step = int(alloc_key.split(".step_")[1].split(".")[0])
                    remote_rank = (rank - step - 1) % world_size
                    recv = torch.cat([k_chunks[remote_rank], v_chunks[remote_rank]], dim=-1)
                    return _AsyncRingExchange(
                        send_tensor=x,
                        recv_tensor=recv,
                        works=[],
                    )

                def fake_dense(q, k, v, *, causal):
                    calls.append((q.size(-2), k.size(-2), causal))
                    return _dense_attention_block(q, k, v, causal=causal)

                layer_functional.dist.get_world_size = fake_get_world_size
                layer_functional.dist.get_rank = fake_get_rank
                layer_functional._ring_exchange_tensor_async = fake_exchange
                layer_functional.flash_attn_dense_with_lse = fake_dense

                out, _ = _flash_ring_forward(
                    q_chunks[rank],
                    k_chunks[rank],
                    v_chunks[rank],
                    group=group,
                    module_id=11,
                    mb_idx=0,
                )

                expected = _global_dense_causal_attention(
                    q_chunks[rank],
                    all_k,
                    all_v,
                    q_positions=positions[rank],
                )
                assert calls == _expected_zigzag_calls(rank=rank, world_size=world_size, local_seq_len=local_seq_len)
                assert torch.allclose(out, expected, atol=1e-5)
    finally:
        layer_functional.dist.get_world_size = original_get_world_size
        layer_functional.dist.get_rank = original_get_rank
        layer_functional._ring_exchange_tensor_async = original_exchange
        layer_functional.flash_attn_dense_with_lse = original_dense


def test_flash_ring_backward_uses_dense_zigzag_half_schedule() -> None:
    batch_size = 1
    num_heads = 1
    local_seq_len = 4
    head_dim = 3

    original_get_world_size = layer_functional.dist.get_world_size
    original_get_rank = layer_functional.dist.get_rank
    original_exchange = layer_functional._ring_exchange_tensor_async
    original_backward = layer_functional.flash_attn_dense_backward
    original_ring_shift = layer_functional.ring_shift
    try:
        for world_size in (2, 4):
            positions = [
                expected_zigzag_ring_positions(local_seq_len, rank=rank, world_size=world_size, device=torch.device("cpu")).view(1, -1)
                for rank in range(world_size)
            ]
            k_chunks = [
                torch.randn((batch_size, num_heads, local_seq_len, head_dim), generator=torch.Generator().manual_seed(10 + rank))
                for rank in range(world_size)
            ]
            v_chunks = [
                torch.randn((batch_size, num_heads, local_seq_len, head_dim), generator=torch.Generator().manual_seed(20 + rank))
                for rank in range(world_size)
            ]
            q_chunks = [
                torch.randn((batch_size, num_heads, local_seq_len, head_dim), generator=torch.Generator().manual_seed(30 + rank))
                for rank in range(world_size)
            ]

            for rank in range(world_size):
                calls: list[tuple[int, int, bool]] = []
                group = _FakeGroup(rank=rank, world_size=world_size)

                def fake_get_world_size(fake_group) -> int:
                    return fake_group.world_size

                def fake_get_rank(fake_group) -> int:
                    return fake_group.rank

                def fake_exchange(x, fake_group, send_to, recv_from, *, alloc_key):
                    if ".kv" in alloc_key:
                        step = int(alloc_key.split(".step_")[1].split(".")[0])
                        remote_rank = (rank - step - 1) % world_size
                        recv = torch.cat([k_chunks[remote_rank], v_chunks[remote_rank]], dim=-1)
                        return _AsyncRingExchange(
                            send_tensor=x,
                            recv_tensor=recv,
                            works=[],
                        )
                    return _AsyncRingExchange(
                        send_tensor=x,
                        recv_tensor=torch.zeros_like(x),
                        works=[],
                    )

                def fake_backward(dout, q, k, v, out, softmax_lse, *, causal):
                    calls.append((q.size(-2), k.size(-2), causal))
                    return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)

                def fake_ring_shift(x, group, send_to, recv_from, *, alloc_key):
                    return x

                layer_functional.dist.get_world_size = fake_get_world_size
                layer_functional.dist.get_rank = fake_get_rank
                layer_functional._ring_exchange_tensor_async = fake_exchange
                layer_functional.flash_attn_dense_backward = fake_backward
                layer_functional.ring_shift = fake_ring_shift

                dq, dk, dv = _flash_ring_backward(
                    torch.randn_like(q_chunks[rank]),
                    q_chunks[rank],
                    k_chunks[rank],
                    v_chunks[rank],
                    torch.randn_like(q_chunks[rank]),
                    torch.randn((batch_size, num_heads, local_seq_len)),
                    group=group,
                    module_id=11,
                    mb_idx=0,
                )

                assert calls == _expected_zigzag_calls(rank=rank, world_size=world_size, local_seq_len=local_seq_len)
                assert dq.shape == q_chunks[rank].shape
                assert dk.shape == k_chunks[rank].shape
                assert dv.shape == v_chunks[rank].shape
    finally:
        layer_functional.dist.get_world_size = original_get_world_size
        layer_functional.dist.get_rank = original_get_rank
        layer_functional._ring_exchange_tensor_async = original_exchange
        layer_functional.flash_attn_dense_backward = original_backward
        layer_functional.ring_shift = original_ring_shift


def main() -> None:
    test_local_attention_masks_cross_example_tokens()
    test_llama_attention_core_masks_cross_example_tokens()
    test_cp_attention_masks_cross_example_tokens()
    test_pack_varlen_qkv_builds_expected_cu_seqlens()
    test_pack_varlen_prefix_qkv_builds_expected_qk_segments()
    test_pack_varlen_prefix_qkv_can_skip_empty_prefix_runs()
    test_expected_zigzag_ring_positions_match_cp_layout()
    test_pack_dense_varlen_qkv_builds_uniform_cu_seqlens()
    test_flash_ring_forward_uses_dense_zigzag_half_schedule()
    test_flash_ring_backward_uses_dense_zigzag_half_schedule()
    print("sequence_ids attention mask ok")


if __name__ == "__main__":
    main()
