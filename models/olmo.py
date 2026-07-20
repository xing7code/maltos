from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.activation_checkpointing import ActivationCheckpointConfig
from models.llama import LlamaMLP, LlamaRMSNorm, LlamaRotaryEmbedding, _LlamaAttentionCore, _apply_rotary, _repeat_kv
from parallel.specs import ContextParallelSpec
from parallel.specs import PipelineParallelSpec
from parallel.specs import TpSpParallelSpec, TpSpShardAxis, TpSpShardRule
from runtime.layers.distributed_rmsnorm import DistributedRMSNorm
from utils.attention_backend import AttentionBackend, validate_attention_backend
from utils.activation_checkpoint import activation_checkpoint_context_fn
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


@dataclass(frozen=True)
class OlmoConfig:
    vocab_size: int = 100352
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_backend: str = AttentionBackend.SDPA_AUTO
    activation_checkpointing: ActivationCheckpointConfig = field(default_factory=ActivationCheckpointConfig)

    def __post_init__(self) -> None:
        validate_attention_backend(self.attention_backend)


class OlmoRMSNorm(LlamaRMSNorm):
    pass


class OlmoAttention(nn.Module):
    def __init__(self, config: OlmoConfig) -> None:
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads or config.num_attention_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = DistributedRMSNorm(self.num_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = DistributedRMSNorm(self.num_kv_heads * self.head_dim, config.rms_norm_eps)
        self.attn_core = _LlamaAttentionCore(config.attention_backend)

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_offset: int = 0,
        position_ids: torch.Tensor | None = None,
        sequence_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        if position_ids is None:
            cos, sin = self.rotary_emb(position_offset, position_offset + seq_len)
        else:
            cos, sin = self.rotary_emb(position_ids=position_ids)
        cos = cos.to(device=x.device, dtype=x.dtype)
        sin = sin.to(device=x.device, dtype=x.dtype)
        q = self.q_norm(self.q_proj(x)).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_norm(self.k_proj(x)).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        repeats = q.size(1) // k.size(1)
        k = _repeat_kv(k, repeats)
        v = _repeat_kv(v, repeats)
        vo = self.attn_core(q, k, v, position_offset, position_ids, sequence_ids)
        out = vo.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class OlmoDecoderLayer(nn.Module):
    def __init__(self, config: OlmoConfig) -> None:
        super().__init__()
        self.self_attn = OlmoAttention(config)
        self.mlp = LlamaMLP(config)
        self.post_attention_layernorm = OlmoRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = OlmoRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
        position_ids: torch.Tensor | None = None,
        sequence_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.self_attn(
            x,
            position_offset=position_offset,
            position_ids=position_ids,
            sequence_ids=sequence_ids,
        )
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x
        return x


class OlmoForCausalLM(nn.Module):
    def __init__(self, config: OlmoConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([OlmoDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = OlmoRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

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
                raise ValueError("OlmoForCausalLM requires input_ids or hidden_states")
            if self.embed_tokens is None:
                raise ValueError("OlmoForCausalLM PP non-first stage requires hidden_states input")
            x = self.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(self.layers):
            if self.training and self.config.activation_checkpointing.should_checkpoint_layer(layer_idx):
                x = checkpoint(
                    lambda y: layer(
                        y,
                        position_offset=position_offset,
                        position_ids=position_ids,
                        sequence_ids=sequence_ids,
                    ),
                    x,
                    use_reentrant=False,
                    context_fn=activation_checkpoint_context_fn,
                )
            else:
                x = layer(x, position_offset=position_offset, position_ids=position_ids, sequence_ids=sequence_ids)
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
            ignore_index=IGNORE_INDEX,
        )
        if loss_weight is not None:
            loss = loss * float(loss_weight)
        return loss

    def pipeline_parallel_spec(self) -> PipelineParallelSpec:
        return PipelineParallelSpec(
            head_layers=["embed_tokens"],
            pipe_layers=["layers"],
            tail_layers=["norm", "lm_head"],
        )

    def context_parallel_spec(self) -> ContextParallelSpec:
        return ContextParallelSpec(
            attention_paths=[f"layers.{i}.self_attn" for i in range(len(self.layers))],
        )

    def flops_per_token(self) -> float:
        hidden = self.config.hidden_size
        intermediate = self.config.intermediate_size
        seq_len = self.config.max_position_embeddings
        vocab = self.config.vocab_size
        qkv_proj = 6 * hidden * hidden
        qk_norm = 4 * hidden
        out_proj = 2 * hidden * hidden
        mlp = 6 * hidden * intermediate
        attention = 4 * seq_len * hidden
        logits = 2 * hidden * vocab
        forward_flops = self.config.num_hidden_layers * (qkv_proj + qk_norm + out_proj + mlp + attention) + logits
        return float(3 * forward_flops)


class OlmoForCausalLMTp(OlmoForCausalLM):
    def tpsp_parallelize_spec(self) -> TpSpParallelSpec:
        rules = []
        for i in range(len(self.layers)):
            rules += [
                TpSpShardRule(f"layers.{i}.self_attn.q_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.self_attn.k_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.self_attn.v_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.self_attn.q_norm", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.self_attn.k_norm", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.self_attn.o_proj", shard_axis=TpSpShardAxis.PARAM_IN, post_comm="all_reduce"),
                TpSpShardRule(f"layers.{i}.mlp.gate_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.mlp.up_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.mlp.down_proj", shard_axis=TpSpShardAxis.PARAM_IN, post_comm="all_reduce"),
            ]
        return TpSpParallelSpec(rules=rules, tie_rules=[])


class OlmoForCausalLMTpSp(OlmoForCausalLM):
    def tpsp_parallelize_spec(self) -> TpSpParallelSpec:
        rules = [
            TpSpShardRule("embed_tokens", shard_axis=TpSpShardAxis.SEQUENCE, post_comm="scatter", comm_dim=1),
        ]
        for i in range(len(self.layers)):
            rules += [
                TpSpShardRule(f"layers.{i}.self_attn", shard_axis=TpSpShardAxis.SEQUENCE, pre_comm="all_gather", comm_dim=1),
                TpSpShardRule(f"layers.{i}.self_attn.q_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.self_attn.k_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.self_attn.v_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.self_attn.q_norm", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.self_attn.k_norm", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.self_attn.o_proj", shard_axis=TpSpShardAxis.PARAM_IN, post_comm="reduce_scatter"),
                TpSpShardRule(f"layers.{i}.mlp", shard_axis=TpSpShardAxis.SEQUENCE, pre_comm="all_gather", comm_dim=1),
                TpSpShardRule(f"layers.{i}.mlp.gate_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.mlp.up_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.mlp.down_proj", shard_axis=TpSpShardAxis.PARAM_IN, post_comm="reduce_scatter"),
            ]
        rules += [
            TpSpShardRule("lm_head", shard_axis=TpSpShardAxis.SEQUENCE, pre_comm="all_gather", comm_dim=1),
        ]
        return TpSpParallelSpec(rules=rules, tie_rules=[])
