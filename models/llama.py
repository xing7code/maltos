from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from models.activation_checkpointing import ActivationCheckpointConfig
from parallel.specs import ContextParallelSpec
from parallel.specs import PipelineParallelSpec
from parallel.specs import TpSpParallelSpec, TpSpShardAxis, TpSpShardRule
from utils.constants import HIDDEN_STATES_KEY, IGNORE_INDEX, INPUT_IDS_KEY, LABELS_KEY, LOSS_WEIGHT_KEY, POSITION_IDS_KEY, POSITION_OFFSET_KEY


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int | None = None
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    attention_backend: str = "sdpa_auto"
    activation_checkpointing: ActivationCheckpointConfig = field(default_factory=ActivationCheckpointConfig)

    def __post_init__(self) -> None:
        valid_backends = {"eager", "sdpa_auto", "sdpa_flash"}
        if self.attention_backend not in valid_backends:
            raise ValueError(f"attention_backend must be one of {sorted(valid_backends)}, got {self.attention_backend!r}")


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + self.eps)).to(dtype=x.dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(
        self,
        start: int | None = None,
        end: int | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids is not None:
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            flat = position_ids.to(device=self.cos.device, dtype=torch.long).reshape(-1)
            cos = self.cos.index_select(0, flat).view(*position_ids.shape, -1).unsqueeze(1)
            sin = self.sin.index_select(0, flat).view(*position_ids.shape, -1).unsqueeze(1)
            return cos, sin
        if start is None or end is None:
            raise ValueError("LlamaRotaryEmbedding requires either position_ids or start/end")
        cos = self.cos[start:end].unsqueeze(0).unsqueeze(0)
        sin = self.sin[start:end].unsqueeze(0).unsqueeze(0)
        return cos, sin


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


def _repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)




def _eager_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    seq_len = q.size(-2)
    causal_mask = torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
    probs = scores.softmax(dim=-1)
    return torch.matmul(probs, v)


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
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
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attn_core = _LlamaAttentionCore(config.attention_backend)

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_offset: int = 0,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        if position_ids is None:
            cos, sin = self.rotary_emb(position_offset, position_offset + seq_len)
        else:
            cos, sin = self.rotary_emb(position_ids=position_ids)
        cos = cos.to(device=x.device, dtype=x.dtype)
        sin = sin.to(device=x.device, dtype=x.dtype)
        q = self.q_proj(x).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        repeats = q.size(1) // k.size(1)
        k = _repeat_kv(k, repeats)
        v = _repeat_kv(v, repeats)
        vo = self.attn_core(q, k, v, position_offset, position_ids)
        out = vo.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class _LlamaAttentionCore(nn.Module):
    def __init__(self, attention_backend: str) -> None:
        super().__init__()
        self.attention_backend = attention_backend

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_offset: int,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del position_offset, position_ids
        if self.attention_backend == "sdpa_auto":
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.attention_backend == "sdpa_flash":
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return _eager_causal_attention(q, k, v)


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), position_offset=position_offset, position_ids=position_ids)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
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
            loss_weight = batch.get(LOSS_WEIGHT_KEY)
        elif isinstance(batch, (tuple, list)):
            input_ids, labels = batch
            hidden_states = None
            position_offset = 0
            position_ids = None
            loss_weight = None
        else:
            input_ids, labels = batch, None
            hidden_states = None
            position_offset = 0
            position_ids = None
            loss_weight = None

        if hidden_states is not None:
            x = hidden_states
        else:
            if input_ids is None:
                raise ValueError("LlamaForCausalLM requires input_ids or hidden_states")
            if self.embed_tokens is None:
                raise ValueError("LlamaForCausalLM PP non-first stage requires hidden_states input")
            x = self.embed_tokens(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            if self.training and self.config.activation_checkpointing.should_checkpoint_layer(layer_idx):
                x = checkpoint(
                    lambda y: layer(y, position_offset=position_offset, position_ids=position_ids),
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x, position_offset=position_offset, position_ids=position_ids)
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
        out_proj = 2 * hidden * hidden
        mlp = 6 * hidden * intermediate
        attention = 4 * seq_len * hidden
        logits = 2 * hidden * vocab
        forward_flops = self.config.num_hidden_layers * (qkv_proj + out_proj + mlp + attention) + logits
        return float(3 * forward_flops)


class LlamaForCausalLMTp(LlamaForCausalLM):
    def tpsp_parallelize_spec(self) -> TpSpParallelSpec:
        rules = []
        for i in range(len(self.layers)):
            rules += [
                TpSpShardRule(f"layers.{i}.self_attn.q_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.self_attn.k_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.self_attn.v_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.self_attn.o_proj", shard_axis=TpSpShardAxis.PARAM_IN, post_comm="all_reduce"),
                TpSpShardRule(f"layers.{i}.mlp.gate_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.mlp.up_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.mlp.down_proj", shard_axis=TpSpShardAxis.PARAM_IN, post_comm="all_reduce"),
            ]
        return TpSpParallelSpec(rules=rules, tie_rules=[])


class LlamaForCausalLMTpSp(LlamaForCausalLM):
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
