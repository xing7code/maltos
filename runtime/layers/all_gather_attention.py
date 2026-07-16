from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from runtime.layers.attn_masking_utils import canonical_position_ids, canonical_sequence_ids
from runtime.layers.functional import all_gather
from utils.attention_backend import AttentionBackend, causal_attention, validate_attention_backend


class AllGatherKvAttentionCore(nn.Module):
    def __init__(self, group: dist.ProcessGroup, attention_backend: str = AttentionBackend.EAGER) -> None:
        super().__init__()
        self.group = group
        self.attention_backend = validate_attention_backend(attention_backend)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_offset: int,
        position_ids: torch.Tensor | None = None,
        sequence_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gathered_k = all_gather(
            k,
            self.group,
            comm_dim=2,
            alloc_key=f"cp.all_gather_kv.{id(self)}.k",
            backward_reduce_op=dist.ReduceOp.SUM,
        )
        gathered_v = all_gather(
            v,
            self.group,
            comm_dim=2,
            alloc_key=f"cp.all_gather_kv.{id(self)}.v",
            backward_reduce_op=dist.ReduceOp.SUM,
        )
        local_positions = canonical_position_ids(
            position_ids,
            batch_size=q.size(0),
            seq_len=q.size(-2),
            position_offset=position_offset,
            device=q.device,
        )
        local_sequence_ids = canonical_sequence_ids(
            sequence_ids,
            batch_size=q.size(0),
            seq_len=q.size(-2),
            device=q.device,
        )
        gathered_positions = [torch.empty_like(local_positions) for _ in range(dist.get_world_size(self.group))]
        dist.all_gather(gathered_positions, local_positions.contiguous(), group=self.group)
        gathered_sequence_ids = None
        if local_sequence_ids is not None:
            gathered_sequence_ids = [torch.empty_like(local_sequence_ids) for _ in range(dist.get_world_size(self.group))]
            dist.all_gather(gathered_sequence_ids, local_sequence_ids.contiguous(), group=self.group)
        return causal_attention(
            q,
            gathered_k,
            gathered_v,
            attention_backend=self.attention_backend,
            warning_prefix="AllGatherKvAttentionCore",
            q_positions=local_positions,
            k_positions=torch.cat(gathered_positions, dim=1),
            q_sequence_ids=local_sequence_ids,
            k_sequence_ids=None if gathered_sequence_ids is None else torch.cat(gathered_sequence_ids, dim=1),
            flash_varlen_mode="prefix",
        )
