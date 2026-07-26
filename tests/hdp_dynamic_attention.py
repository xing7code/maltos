"""Distributed numerical contract for the HDP-balanced dynamic sub-ring.

Usage:
  PYTHONPATH=. .venv/bin/python tests/hdp_dynamic_attention.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from data.bytescale_hdp import schedule_document_waves
from parallel.hdp_helper import ByteScaleHdpBalancedConfig, DocumentIndices
from runtime.layers.hdp_attention import HdpBalancedAttentionCore, _single_document_attention


_PORT = 29682
_ATOL = 2e-5


def _run_worker(rank: int) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{_PORT}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.manual_seed(7)
        full_q = torch.randn(1, 4, 8, 4, requires_grad=True)
        full_k = torch.randn(1, 2, 8, 4, requires_grad=True)
        full_v = torch.randn(1, 2, 8, 4, requires_grad=True)
        short_q = torch.randn(1, 4, 2, 4, requires_grad=True)
        short_k = torch.randn(1, 2, 2, 4, requires_grad=True)
        short_v = torch.randn(1, 2, 2, 4, requires_grad=True)
        positions = torch.arange(8, dtype=torch.long).unsqueeze(0)
        reference_out = _single_document_attention(
            full_q,
            full_k,
            full_v,
            positions=positions,
            sequence_ids=None,
        )
        short_reference = _single_document_attention(
            short_q,
            short_k,
            short_v,
            positions=torch.arange(2).unsqueeze(0),
            sequence_ids=None,
        )
        (reference_out.sum() + short_reference.sum()).backward()

        indices = (DocumentIndices(0, tuple(range(8))),)
        layout = schedule_document_waves(
            indices,
            rank=rank,
            world_size=2,
            config=ByteScaleHdpBalancedConfig(partition_tokens=6),
            global_valid_targets=10,
        )[0]
        document = layout.documents[0]
        local_positions = torch.tensor(
            [index for index in document.source_indices if index is not None],
            dtype=torch.long,
        )
        q = torch.zeros(1, 4, 6, 4)
        k = torch.zeros(1, 2, 6, 4)
        v = torch.zeros(1, 2, 6, 4)
        q[:, :, :4, :] = full_q.detach().index_select(2, local_positions)
        k[:, :, :4, :] = full_k.detach().index_select(2, local_positions)
        v[:, :, :4, :] = full_v.detach().index_select(2, local_positions)
        assert layout.document_ids == (0,)
        q = q.requires_grad_(True)
        k = k.requires_grad_(True)
        v = v.requires_grad_(True)
        core = HdpBalancedAttentionCore(dist.group.WORLD)
        core.set_active_schedule(layout)
        local_position_ids = torch.zeros((1, 6), dtype=torch.long)
        local_position_ids[:, :4] = local_positions
        out = core(q, k, v, position_offset=0, position_ids=local_position_ids)
        out.sum().backward()

        reference_out_local = torch.zeros_like(out)
        reference_out_local[:, :, :4, :] = reference_out.detach().index_select(
            2,
            local_positions,
        )
        reference_q_grad = torch.zeros_like(q)
        reference_k_grad = torch.zeros_like(k)
        reference_v_grad = torch.zeros_like(v)
        reference_q_grad[:, :, :4, :] = full_q.grad.detach().index_select(
            2,
            local_positions,
        )
        reference_k_grad[:, :, :4, :] = full_k.grad.detach().index_select(
            2,
            local_positions,
        )
        reference_v_grad[:, :, :4, :] = full_v.grad.detach().index_select(
            2,
            local_positions,
        )
        worst = torch.stack(
            (
                (out.detach() - reference_out_local).abs().max(),
                (q.grad - reference_q_grad).abs().max(),
                (k.grad - reference_k_grad).abs().max(),
                (v.grad - reference_v_grad).abs().max(),
            )
        ).max()
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"HDP dynamic-ring max diff: {worst.item():.2e} (atol={_ATOL:.2e})")
            if worst.item() > _ATOL:
                raise AssertionError(f"HDP dynamic ring mismatch: {worst.item():.2e}")
            print("PASS")
    finally:
        dist.destroy_process_group()


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, nprocs=2, join=True)


if __name__ == "__main__":
    main()
