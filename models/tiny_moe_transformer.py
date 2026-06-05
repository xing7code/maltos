from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from parallel.context import ContextParallelSpec
from parallel.expert import ExpertParallelMoEModule, ExpertParallelSpec
from parallel.pipeline import PipelineParallelSpec
from parallel.specs import TpSpComm, TpSpParallelSpec, TpSpShardAxis, TpSpShardRule
from models.tiny_transformer import CausalSelfAttention, RmsNorm, RoPE, MLP


class Top1MoE(nn.Module):
    def __init__(self, dim: int, hidden_size: int, num_experts: int) -> None:
        super().__init__()
        self._dim = dim
        self._hidden_size = hidden_size
        self._num_experts = num_experts
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(dim, hidden_size) for _ in range(num_experts)])

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_experts(self) -> int:
        return self._num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        flat = x.reshape(-1, hidden)
        router_logits = self.router(flat)
        router_probs = router_logits.softmax(dim=-1)
        expert_idx = router_probs.argmax(dim=-1)
        expert_weight = router_probs.gather(1, expert_idx.unsqueeze(1)).squeeze(1)
        out = torch.zeros_like(flat)
        for idx, expert in enumerate(self.experts):
            mask = expert_idx == idx
            if not torch.any(mask):
                continue
            out[mask] = expert(flat[mask]) * expert_weight[mask].unsqueeze(1)
        return out.view(batch, seq_len, hidden)


class MoETransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, hidden_size: int, eps: float, num_experts: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, n_heads, n_kv_heads)
        self.norm1 = RmsNorm(dim, eps)
        self.moe = Top1MoE(dim, hidden_size, num_experts)
        self.norm2 = RmsNorm(dim, eps)

    def forward(self, x: torch.Tensor, cos=None, sin=None, position_offset: int = 0) -> torch.Tensor:
        x = self.attn(self.norm1(x), cos, sin, position_offset=position_offset) + x
        x = self.moe(self.norm2(x)) + x
        return x


class TinyMoETransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        hidden_size: int,
        eps: float,
        n_layers: int,
        vocab_size: int,
        max_seq_len: int,
        num_experts: int,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.rope = RoPE(dim // n_heads, max_seq_len)
        self.layers = nn.ModuleList(
            [
                MoETransformerBlock(dim, n_heads, n_kv_heads, hidden_size, eps, num_experts)
                for _ in range(n_layers)
            ]
        )
        self.norm = RmsNorm(dim, eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, batch):
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids")
            hidden_states = batch.get("hidden_states")
            labels = batch.get("labels")
            position_offset = int(batch.get("position_offset", 0))
            loss_weight = batch.get("loss_weight")
        elif isinstance(batch, (tuple, list)):
            input_ids, labels = batch
            hidden_states = None
            position_offset = 0
            loss_weight = None
        else:
            input_ids, labels = batch, None
            hidden_states = None
            position_offset = 0
            loss_weight = None

        if hidden_states is not None:
            x = hidden_states
        else:
            if input_ids is None:
                raise ValueError("TinyMoETransformer requires input_ids or hidden_states")
            if self.embed is None:
                raise ValueError("TinyMoETransformer PP non-first stage requires hidden_states input")
            x = self.embed(input_ids)

        _, seq_len, _ = x.shape
        cos, sin = self.rope(position_offset, position_offset + seq_len)
        for layer in self.layers:
            x = layer(x, cos, sin, position_offset=position_offset)
        if self.norm is None or self.lm_head is None:
            return x
        logits = self.lm_head(self.norm(x))
        if labels is None:
            return logits
        if input_ids is not None and labels.shape != input_ids.shape:
            raise ValueError(f"labels shape must match input_ids shape, got {labels.shape} vs {input_ids.shape}")
        loss = F.cross_entropy(
            logits.contiguous().view(-1, logits.size(-1)),
            labels.contiguous().view(-1),
            ignore_index=-100,
        )
        if loss_weight is not None:
            loss = loss * float(loss_weight)
        return loss

    def expert_parallel_spec(self) -> ExpertParallelSpec:
        return ExpertParallelSpec(
            moe_paths=[f"layers.{i}.moe" for i in range(len(self.layers))],
        )

    def pipeline_parallel_spec(self) -> PipelineParallelSpec:
        return PipelineParallelSpec(
            head_layers=["embed"],
            pipe_layers=["layers"],
            tail_layers=["norm", "lm_head"],
        )

    def context_parallel_spec(self) -> ContextParallelSpec:
        return ContextParallelSpec(
            attention_paths=[f"layers.{i}.attn" for i in range(len(self.layers))],
        )


class TinyMoETransformerTp(TinyMoETransformer):
    def tpsp_parallelize_spec(self) -> TpSpParallelSpec:
        rules = []
        for i in range(len(self.layers)):
            rules += [
                TpSpShardRule(f"layers.{i}.attn.q_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.attn.kv_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.attn.o_proj", shard_axis=TpSpShardAxis.PARAM_IN, post_comm="all_reduce"),
            ]
        return TpSpParallelSpec(rules=rules, tie_rules=[])


class TinyMoETransformerTpSp(TinyMoETransformer):
    def tpsp_parallelize_spec(self) -> TpSpParallelSpec:
        rules = [
            TpSpShardRule("embed", shard_axis=TpSpShardAxis.SEQUENCE, post_comm=TpSpComm.SCATTER, comm_dim=1),
        ]
        for i in range(len(self.layers)):
            rules += [
                TpSpShardRule(
                    f"layers.{i}.attn",
                    shard_axis=TpSpShardAxis.SEQUENCE,
                    pre_comm=TpSpComm.ALL_GATHER,
                    comm_dim=1,
                ),
                TpSpShardRule(f"layers.{i}.attn.q_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.attn.kv_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(
                    f"layers.{i}.attn.o_proj",
                    shard_axis=TpSpShardAxis.PARAM_IN,
                    post_comm=TpSpComm.REDUCE_SCATTER,
                ),
            ]
        rules += [
            TpSpShardRule("lm_head", shard_axis=TpSpShardAxis.SEQUENCE, pre_comm=TpSpComm.ALL_GATHER, comm_dim=1),
        ]
        return TpSpParallelSpec(rules=rules, tie_rules=[])


ExpertParallelMoEModule.register(Top1MoE)
