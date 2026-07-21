from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from parallel.expert_interfaces import ExpertParallelMoEModule
from parallel.specs import ContextParallelSpec, ExpertParallelSpec, PipelineParallelSpec
from parallel.specs import TpSpComm, TpSpParallelSpec, TpSpShardAxis, TpSpShardRule
from runtime.types import LossOutput, PipelineOutput
from models.tiny_transformer import CausalSelfAttention, RmsNorm, RoPE, MLP
from utils.attention_backend import AttentionBackend
from utils.constants import (
    HIDDEN_STATES_KEY,
    IGNORE_INDEX,
    INPUT_IDS_KEY,
    LABELS_KEY,
    LOSS_WEIGHT_KEY,
    POSITION_IDS_KEY,
    POSITION_OFFSET_KEY,
    SEQUENCE_IDS_KEY,
)


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

    def forward(self, x: torch.Tensor, *, return_aux_loss: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        output = out.view(batch, seq_len, hidden)
        if not return_aux_loss:
            return output
        aux_loss = _top1_load_balance_loss(router_logits, expert_idx)
        return output, aux_loss


class MoETransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        hidden_size: int,
        eps: float,
        num_experts: int,
        attention_backend: str = AttentionBackend.EAGER,
    ):
        super().__init__()
        self.attn = CausalSelfAttention(dim, n_heads, n_kv_heads, attention_backend=attention_backend)
        self.norm1 = RmsNorm(dim, eps)
        self.moe = Top1MoE(dim, hidden_size, num_experts)
        self.norm2 = RmsNorm(dim, eps)

    def forward(
        self,
        x: torch.Tensor,
        cos=None,
        sin=None,
        position_offset: int = 0,
        position_ids: torch.Tensor | None = None,
        sequence_ids: torch.Tensor | None = None,
        return_aux_loss: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.attn(
            self.norm1(x),
            cos,
            sin,
            position_offset=position_offset,
            position_ids=position_ids,
            sequence_ids=sequence_ids,
        ) + x
        moe_out = self.moe(self.norm2(x), return_aux_loss=return_aux_loss)
        if not return_aux_loss:
            return moe_out + x
        moe_output, aux_loss = moe_out
        return moe_output + x, aux_loss


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
        moe_aux_loss_coef: float = 0.0,
        attention_backend: str = AttentionBackend.EAGER,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.rope = RoPE(dim // n_heads, max_seq_len)
        self.layers = nn.ModuleList(
            [
                MoETransformerBlock(
                    dim,
                    n_heads,
                    n_kv_heads,
                    hidden_size,
                    eps,
                    num_experts,
                    attention_backend=attention_backend,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RmsNorm(dim, eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if moe_aux_loss_coef < 0:
            raise ValueError("moe_aux_loss_coef must be >= 0")
        self.moe_aux_loss_coef = float(moe_aux_loss_coef)
        self._total_moe_layers = n_layers

    def forward(self, batch):
        if isinstance(batch, dict):
            input_ids = batch.get(INPUT_IDS_KEY)
            hidden_states = batch.get(HIDDEN_STATES_KEY)
            labels = batch.get(LABELS_KEY)
            position_offset = int(batch.get(POSITION_OFFSET_KEY, 0))
            position_ids = batch.get(POSITION_IDS_KEY)
            sequence_ids = batch.get(SEQUENCE_IDS_KEY)
            loss_weight = batch.get(LOSS_WEIGHT_KEY)
        elif isinstance(batch, (tuple, list)):
            input_ids, labels = batch
            hidden_states = None
            position_offset = 0
            position_ids = None
            sequence_ids = None
            loss_weight = None
        else:
            input_ids, labels = batch, None
            hidden_states = None
            position_offset = 0
            position_ids = None
            sequence_ids = None
            loss_weight = None

        if hidden_states is not None:
            x = hidden_states
        else:
            if input_ids is None:
                raise ValueError("TinyMoETransformer requires input_ids or hidden_states")
            if self.embed is None:
                raise ValueError("TinyMoETransformer PP non-first stage requires hidden_states input")
            x = self.embed(input_ids)

        batch_size, seq_len, _ = x.shape
        if position_ids is None:
            cos, sin = self.rope(position_offset, position_offset + seq_len)
        else:
            cos, sin = self.rope(position_ids=position_ids)
        aux_losses: list[torch.Tensor] = []
        collect_aux_loss = self.moe_aux_loss_coef > 0
        for layer in self.layers:
            layer_out = layer(
                x,
                cos,
                sin,
                position_offset=position_offset,
                position_ids=position_ids,
                sequence_ids=sequence_ids,
                return_aux_loss=collect_aux_loss,
            )
            if collect_aux_loss:
                if isinstance(layer_out, tuple):
                    x, aux_loss = layer_out
                    aux_losses.append(aux_loss)
                else:
                    # PP replaces layers owned by another stage with identity
                    # modules. They intentionally carry no router loss.
                    x = layer_out
            else:
                x = layer_out
        if collect_aux_loss:
            balance_loss = (
                torch.stack(aux_losses).sum() / float(self._total_moe_layers)
                if aux_losses
                else x.new_zeros(())
            )
            auxiliary_loss = self.moe_aux_loss_coef * balance_loss
        if self.norm is None or self.lm_head is None:
            if collect_aux_loss:
                return PipelineOutput(activation=x, auxiliary_loss=auxiliary_loss)
            return x
        logits = self.lm_head(self.norm(x))
        if labels is None:
            return logits
        if input_ids is not None and labels.shape != input_ids.shape:
            raise ValueError(f"labels shape must match input_ids shape, got {labels.shape} vs {input_ids.shape}")
        loss = F.cross_entropy(
            logits.contiguous().view(-1, logits.size(-1)),
            labels.contiguous().view(-1),
            ignore_index=IGNORE_INDEX,
        )
        if loss_weight is not None:
            loss = loss * float(loss_weight)
        if not collect_aux_loss:
            return loss
        total_loss = loss + auxiliary_loss
        return LossOutput(
            loss=total_loss,
            metrics={
                "loss/ce": float(loss.detach().float().item()),
                "moe/load_balance_loss": float(balance_loss.detach().float().item()),
                "loss/total": float(total_loss.detach().float().item()),
            },
            auxiliary_loss=auxiliary_loss,
        )

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
                TpSpShardRule(f"layers.{i}.attn.k_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.attn.v_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
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
                TpSpShardRule(f"layers.{i}.attn.k_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.attn.v_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
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


def _top1_load_balance_loss(router_logits: torch.Tensor, expert_idx: torch.Tensor) -> torch.Tensor:
    """Switch-style top-1 router balance loss for one MoE layer.

    The hard assignment fraction is deliberately detached; gradients flow only
    through the mean router probabilities back into the router weights.
    """
    num_experts = router_logits.size(-1)
    router_probs = router_logits.float().softmax(dim=-1)
    expert_fraction = torch.nn.functional.one_hot(expert_idx, num_classes=num_experts).float().mean(dim=0).detach()
    mean_router_prob = router_probs.mean(dim=0)
    return num_experts * torch.sum(expert_fraction * mean_router_prob)
