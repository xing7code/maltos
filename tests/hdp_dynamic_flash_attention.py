"""CUDA/NCCL correctness coverage for HDP's synchronous dynamic Flash Ring.

Run with ``PYTHONPATH=. .venv/bin/python tests/hdp_dynamic_flash_attention.py``.
It deliberately exercises parent-group subsets (including (0, 2)); it never
creates a process group for a document.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import runtime.layers.cp_functional as cp_functional
import runtime.layers.hdp_attention as hdp_attention
from data.bytescale_hdp import schedule_document_waves
from runtime.layers.flash_utils import flash_attn_block_fallback_reason
from runtime.layers.hdp_attention import HdpBalancedAttentionCore, _expand_kv_for_query_heads, _single_document_attention
from runtime.layers.cp_functional import dynamic_flash_ring_attention
from runtime.layers.attn_masking_utils import build_example_causal_mask
from parallel.hdp_helper import ByteScaleHdpBalancedConfig, DocumentIndices
from utils.attention_backend import eager_causal_attention


_PORT = 29683
_ATOL = 3e-2


def _check_local_documents_are_fused(rank: int) -> None:
    """Two D=1 documents must become one Flash varlen invocation, not two."""
    if rank != 0:
        return
    layout = schedule_document_waves(
        (DocumentIndices(0, (0, 1)), DocumentIndices(0, (0, 1, 2, 3))),
        rank=0,
        world_size=1,
        config=ByteScaleHdpBalancedConfig(partition_tokens=8),
        global_valid_targets=6,
    )[0]
    device = torch.device("cuda", rank)
    q = torch.randn(1, 4, 8, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 2, 8, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 2, 8, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], device=device)
    sequence_ids = torch.tensor([[1, 1, 1, 1, 0, 0, -1, -1]], device=device)
    core = HdpBalancedAttentionCore(dist.group.WORLD, attention_backend="flash_attn")
    core.set_active_schedule(layout)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    eager_core = HdpBalancedAttentionCore(dist.group.WORLD, attention_backend="eager")
    eager_core.set_active_schedule(layout)
    eager_out = eager_core(
        q_ref, k_ref, v_ref,
        position_offset=0, position_ids=position_ids, sequence_ids=sequence_ids,
    )
    eager_out.float().sum().backward()
    original = hdp_attention.flash_attn_varlen_segments
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    hdp_attention.flash_attn_varlen_segments = counted
    try:
        flash_out = core(
            q, k, v,
            position_offset=0, position_ids=position_ids, sequence_ids=sequence_ids,
        )
        flash_out.float().sum().backward()
    finally:
        hdp_attention.flash_attn_varlen_segments = original
    assert calls == 1
    assert ("flash_local_segments", str(device)) in layout.attention_metadata_cache
    torch.testing.assert_close(flash_out, eager_out, atol=_ATOL, rtol=_ATOL)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=_ATOL, rtol=_ATOL)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=_ATOL, rtol=_ATOL)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=_ATOL, rtol=_ATOL)


def _check_document(rank: int, participants: tuple[int, ...], *, packed: bool) -> None:
    """Compare native-GQA Flash Ring output and all three gradients to eager."""
    torch.manual_seed(100 + rank)
    device = torch.device("cuda", rank)
    # Local slots are deliberately not globally contiguous for the (0, 2) case.
    local_positions = torch.tensor(
        [rank * 3, rank * 3 + 2], device=device, dtype=torch.long).unsqueeze(0)
    q = torch.randn(1, 4, 2, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 2, 2, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 2, 2, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    sequence_ids = (torch.tensor([[0, 1]], device=device) if packed else None)

    # Instrumentation is intentionally local: K/V must retain their two native heads.
    seen_kv_heads: list[int] = []
    exchange = cp_functional._subset_ring_exchange
    async_exchange = cp_functional._async_subset_ring_exchange
    def checked_exchange(x, **kwargs):
        if x.dim() == 4:
            seen_kv_heads.append(x.size(1))
        return exchange(x, **kwargs)
    def checked_async_exchange(x, **kwargs):
        if x.dim() == 4:
            seen_kv_heads.append(x.size(1))
        return async_exchange(x, **kwargs)
    out = None
    if rank in participants:
        cp_functional._subset_ring_exchange = checked_exchange
        cp_functional._async_subset_ring_exchange = checked_async_exchange
        try:
            out = dynamic_flash_ring_attention(
                q, k, v, positions=local_positions, sequence_ids=sequence_ids,
                group=dist.group.WORLD, participant_ranks=participants,
                partition_tokens=8,
            )
            out.float().sum().backward()
        finally:
            cp_functional._subset_ring_exchange = exchange
            cp_functional._async_subset_ring_exchange = async_exchange
        assert all(heads == 2 for heads in seen_kv_heads), seen_kv_heads
    # Reconstruct the logical document on every participant to compare output,
    # dq, dk and dv.  This is test-only all-gather, never part of HDP runtime.
    gathered = []
    for local in (q.detach(), k.detach(), v.detach(), local_positions, sequence_ids):
        if local is None:
            gathered.append(None)
            continue
        slots = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(slots, local, group=dist.group.WORLD)
        gathered.append(slots)
    q_all, k_all, v_all, pos_all, seq_all = gathered
    assert q_all is not None and k_all is not None and v_all is not None and pos_all is not None
    q_ref = torch.cat([q_all[i] for i in participants], dim=2).detach().clone().requires_grad_(True)
    k_ref = torch.cat([k_all[i] for i in participants], dim=2).detach().clone().requires_grad_(True)
    v_ref = torch.cat([v_all[i] for i in participants], dim=2).detach().clone().requires_grad_(True)
    positions_ref = torch.cat([pos_all[i] for i in participants], dim=1)
    sequence_ref = None if seq_all is None else torch.cat([seq_all[i] for i in participants], dim=1)
    compute_k, compute_v = _expand_kv_for_query_heads(q_ref, k_ref, v_ref)
    mask = build_example_causal_mask(
        q_positions=positions_ref, k_positions=positions_ref,
        q_sequence_ids=sequence_ref, k_sequence_ids=sequence_ref,
    ).unsqueeze(1)
    ref = eager_causal_attention(q_ref, compute_k, compute_v, mask=mask)
    ref.float().sum().backward()
    if rank in participants:
        assert out is not None
        local_index = participants.index(rank)
        token = slice(local_index * 2, (local_index + 1) * 2)
        torch.testing.assert_close(out, ref[:, :, token, :], atol=_ATOL, rtol=_ATOL)
        torch.testing.assert_close(q.grad, q_ref.grad[:, :, token, :], atol=_ATOL, rtol=_ATOL)
        torch.testing.assert_close(k.grad, k_ref.grad[:, :, token, :], atol=_ATOL, rtol=_ATOL)
        torch.testing.assert_close(v.grad, v_ref.grad[:, :, token, :], atol=_ATOL, rtol=_ATOL)


def _worker(rank: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{_PORT}", rank=rank, world_size=3)
    try:
        # single participant, packed sequence boundary; ranks 1 and 2 are dummy
        _check_document(rank, (0,), packed=True)
        dist.barrier()
        # two-rank non-contiguous subset; rank 1 is a dummy HDP rank.
        _check_document(rank, (0, 2), packed=False)
        dist.barrier()
        # three-rank ordered subset, with native 4Q/2KV GQA.
        _check_document(rank, (0, 1, 2), packed=True)
        dist.barrier()
        # A second document with another subset verifies independent document rings.
        _check_document(rank, (1, 2), packed=False)
        dist.barrier()
        _check_local_documents_are_fused(rank)
        dist.barrier()
        if rank == 0:
            print("PASS: HDP Flash Ring single/2/3-rank, noncontiguous, packed, GQA, dummy coverage")
    finally:
        dist.destroy_process_group()


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        print("SKIP: requires at least three CUDA devices for NCCL HDP Flash Ring coverage")
        return
    probe = torch.empty((1, 4, 2, 16), device="cuda", dtype=torch.bfloat16)
    reason = flash_attn_block_fallback_reason(probe)
    if reason is not None:
        print(f"SKIP: {reason}")
        return
    mp.spawn(_worker, nprocs=3, join=True)


def test_flash_backend_fails_fast_when_kernel_is_unavailable() -> None:
    """Backend selection must not silently turn an explicit Flash request eager."""
    core = HdpBalancedAttentionCore(None, attention_backend="flash_attn")
    assert core.attention_backend.value == "flash_attn"
    probe = torch.empty((1, 4, 2, 16), dtype=torch.float32)
    assert flash_attn_block_fallback_reason(probe) is not None


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
