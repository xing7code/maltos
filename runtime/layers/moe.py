from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn

from parallel.expert_interfaces import ExpertParallelMoEModule


@dataclass(frozen=True)
class _MoEMetadata:
    dim: int
    num_experts: int


class ExpertParallelMoE(nn.Module):
    def __init__(
        self,
        *,
        router: nn.Module,
        local_experts: nn.ModuleList,
        local_expert_ids: list[int],
        metadata: _MoEMetadata,
        ep_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self.router = router
        self.local_experts = local_experts
        self.local_expert_ids = tuple(local_expert_ids)
        self.hidden_size = metadata.dim
        self.num_experts = metadata.num_experts
        self.ep_group = ep_group
        self.ep_rank = dist.get_rank(ep_group)
        self.ep_world_size = dist.get_world_size(ep_group)
        if self.num_experts % self.ep_world_size != 0:
            raise ValueError(
                "ExpertParallelPlugin requires num_experts divisible by ep size, "
                f"got num_experts={self.num_experts}, ep={self.ep_world_size}"
            )
        if len(self.local_expert_ids) * self.ep_world_size != self.num_experts:
            raise ValueError("ExpertParallelPlugin expected evenly sharded local experts")

    @classmethod
    def from_moe(cls, moe: ExpertParallelMoEModule, ep_group: dist.ProcessGroup) -> "ExpertParallelMoE":
        num_experts = int(moe.num_experts)
        experts = moe.experts
        if len(experts) != num_experts:
            raise ValueError(f"ExpertParallelPlugin expected len(experts)==num_experts, got {len(experts)} vs {num_experts}")
        ep_rank = dist.get_rank(ep_group)
        ep_world_size = dist.get_world_size(ep_group)
        experts_per_rank = num_experts // ep_world_size
        start = ep_rank * experts_per_rank
        end = start + experts_per_rank
        local_ids = list(range(start, end))
        local_experts = nn.ModuleList([experts[idx] for idx in local_ids])
        hidden_size = int(moe.hidden_size)
        dim = int(moe.dim)
        if dim <= 0 or hidden_size <= 0:
            raise ValueError("ExpertParallelPlugin requires positive moe.dim and moe.hidden_size")
        return cls(
            router=moe.router,
            local_experts=local_experts,
            local_expert_ids=local_ids,
            metadata=_MoEMetadata(dim=dim, num_experts=num_experts),
            ep_group=ep_group,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = x.shape
        flat = x.reshape(-1, hidden_size)
        router_logits = self.router(flat)
        router_probs = router_logits.softmax(dim=-1)
        expert_idx = router_probs.argmax(dim=-1)
        expert_weight = router_probs.gather(1, expert_idx.unsqueeze(1)).squeeze(1)
        experts_per_rank = len(self.local_expert_ids)
        dest_rank = torch.div(expert_idx, experts_per_rank, rounding_mode="floor")
        local_expert_idx = expert_idx - dest_rank * experts_per_rank

        order = torch.argsort(dest_rank)
        send_tokens = flat.index_select(0, order).contiguous()
        send_weights = expert_weight.index_select(0, order).contiguous()
        send_local_expert_idx = local_expert_idx.index_select(0, order).to(dtype=torch.int64).contiguous()
        send_counts = torch.bincount(dest_rank, minlength=self.ep_world_size).to(dtype=torch.int64)
        recv_counts = _exchange_counts(send_counts, self.ep_group)
        send_split_sizes = send_counts.tolist()
        recv_split_sizes = recv_counts.tolist()

        recv_total = int(recv_counts.sum().item())
        recv_tokens = torch.empty((recv_total, hidden_size), dtype=flat.dtype, device=flat.device)
        recv_weights = torch.empty((recv_total,), dtype=expert_weight.dtype, device=flat.device)
        recv_local_expert_idx = torch.empty((recv_total,), dtype=torch.int64, device=flat.device)

        recv_tokens = funcol.all_to_all_single(
            send_tokens,
            output_split_sizes=recv_split_sizes,
            input_split_sizes=send_split_sizes,
            group=self.ep_group,
        )
        recv_weights = funcol.all_to_all_single(
            send_weights,
            output_split_sizes=recv_split_sizes,
            input_split_sizes=send_split_sizes,
            group=self.ep_group,
        )
        recv_local_expert_idx = funcol.all_to_all_single(
            send_local_expert_idx,
            output_split_sizes=recv_split_sizes,
            input_split_sizes=send_split_sizes,
            group=self.ep_group,
        )

        recv_outputs = torch.zeros_like(recv_tokens)
        for local_idx, expert in enumerate(self.local_experts):
            mask = recv_local_expert_idx == local_idx
            # Keep wrapped expert-module execution order identical across EREP
            # replicas even when this rank routed zero tokens to a given
            # expert. ZeRO3 hooks hang if one replica skips a wrapped module
            # that its peer enters, so we still call the expert on the empty
            # tensor rather than skipping the module entirely.
            expert_input = recv_tokens[mask]
            expert_out = expert(expert_input)
            expert_out = expert_out * recv_weights[mask].unsqueeze(1)
            recv_outputs[mask] = expert_out.to(recv_outputs.dtype)

        returned_outputs = torch.empty_like(send_tokens)
        returned_outputs = funcol.all_to_all_single(
            recv_outputs,
            output_split_sizes=send_split_sizes,
            input_split_sizes=recv_split_sizes,
            group=self.ep_group,
        )

        out = torch.zeros_like(flat)
        out.index_copy_(0, order, returned_outputs)
        return out.view(batch, seq_len, hidden_size)


def _exchange_counts(send_counts: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    recv_counts = torch.empty_like(send_counts)
    split_sizes = [1] * send_counts.numel()
    dist.all_to_all_single(
        recv_counts,
        send_counts.contiguous(),
        output_split_sizes=split_sizes,
        input_split_sizes=split_sizes,
        group=group,
    )
    return recv_counts
