from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from parallel.specs import ContextParallelSpec
from parallel.specs import PipelineParallelSpec
from parallel.specs import TpSpParallelSpec, TpSpShardAxis, TpSpShardRule
from utils.attention_backend import AttentionBackend, causal_attention, validate_attention_backend
from utils.attention_masking import canonical_position_ids, canonical_sequence_ids
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
from utils.logging import debug_log


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, attention_backend: str = AttentionBackend.EAGER):
        super().__init__()
        self.head_dim = dim//n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.head_dim*self.n_kv_heads, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim*self.n_kv_heads, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.attn_core = _LocalCausalAttentionCore(attention_backend)

    def forward(
        self,
        x,
        cos=None,
        sin=None,
        position_offset: int = 0,
        position_ids: torch.Tensor | None = None,
        sequence_ids: torch.Tensor | None = None,
    ):
        debug_log(3, f"layer: {self._get_name()} input shape={x.size()}")
        b, s, d = x.size()
        # use -1 for n_heads, n_kv_heads, this dim can be sharded for TP.
        q = self.q_proj(x).view(b, s, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, -1, self.head_dim).transpose(1, 2)
        if self.n_heads != self.n_kv_heads:
            k = k.repeat_interleave(self.n_heads//self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads//self.n_kv_heads, dim=1)
        vo = self.attn_core(q, k, v, position_offset, position_ids, sequence_ids)
        # use -1 for last dim, it can be different than d for TP.
        out = vo.transpose(1, 2).contiguous().view(b, s, -1)
        output = self.o_proj(out)
        debug_log(3, f"layer: {self._get_name()}, output shape={output.size()}")
        return output


class _LocalCausalAttentionCore(nn.Module):
    def __init__(self, attention_backend: str = AttentionBackend.EAGER) -> None:
        super().__init__()
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
        positions = None
        sequences = None
        if position_ids is not None or sequence_ids is not None:
            seq_len = q.size(-2)
            positions = canonical_position_ids(
                position_ids,
                batch_size=q.size(0),
                seq_len=seq_len,
                position_offset=position_offset,
                device=q.device,
            )
            sequences = canonical_sequence_ids(
                sequence_ids,
                batch_size=q.size(0),
                seq_len=seq_len,
                device=q.device,
            )
        return causal_attention(
            q,
            k,
            v,
            attention_backend=self.attention_backend,
            warning_prefix="TinyTransformer",
            q_positions=positions,
            k_positions=positions,
            q_sequence_ids=sequences,
            k_sequence_ids=sequences,
            flash_varlen_mode="same_length",
        )


class MLP(nn.Module):
    def __init__(self, dim, hidden_size):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_size, bias=False)
        self.up = nn.Linear(dim, hidden_size, bias=False)
        self.down = nn.Linear(hidden_size, dim, bias=False)
    
    def forward(self, x):
        debug_log(3, f"layer: {self._get_name()}, input shape={x.size()}")
        output = self.down(F.silu(self.gate(x)) * self.up(x))
        debug_log(3, f"layer: {self._get_name()}, output shape={output.size()}")
        return output


class RmsNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        debug_log(3, f"layer: {self._get_name()}, input shape={x.size()}")
        output = self.weight * x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        debug_log(3, f"layer: {self._get_name()}, output shape={output.size()}")
        return output


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, hidden_size, eps, attention_backend: str = AttentionBackend.EAGER):
        super().__init__()
        self.attn = CausalSelfAttention(dim, n_heads, n_kv_heads, attention_backend=attention_backend)
        self.norm1 = RmsNorm(dim, eps)
        self.mlp = MLP(dim, hidden_size)
        self.norm2 = RmsNorm(dim, eps)
    
    def forward(
        self,
        x,
        cos=None,
        sin=None,
        position_offset: int = 0,
        position_ids: torch.Tensor | None = None,
        sequence_ids: torch.Tensor | None = None,
    ):
        x = self.attn(
            self.norm1(x),
            cos,
            sin,
            position_offset=position_offset,
            position_ids=position_ids,
            sequence_ids=sequence_ids,
        ) + x
        x = self.mlp(self.norm2(x)) + x
        return x


class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000):
        super().__init__()
        theta = 1./(base**(torch.arange(0, head_dim, 2).float() / head_dim))
        seq = torch.arange(max_seq_len)
        freq = torch.outer(seq, theta)
        self.register_buffer("cos", torch.cos(freq))
        self.register_buffer("sin", torch.sin(freq))

    def forward(
        self,
        start: int | None = None,
        end: int | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        if position_ids is not None:
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            flat = position_ids.to(device=self.cos.device, dtype=torch.long).reshape(-1)
            cos = self.cos.index_select(0, flat).view(*position_ids.shape, -1).unsqueeze(1)
            sin = self.sin.index_select(0, flat).view(*position_ids.shape, -1).unsqueeze(1)
            return cos, sin
        if start is None or end is None:
            raise ValueError("RoPE requires either position_ids or start/end")
        return self.cos[start:end].unsqueeze(0).unsqueeze(0), self.sin[start:end].unsqueeze(0).unsqueeze(0)



class TinyTransformer(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        n_kv_heads,
        hidden_size,
        eps,
        n_layers,
        vocab_size,
        max_seq_len,
        attention_backend: str = AttentionBackend.EAGER,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.rope = RoPE(dim//n_heads, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim,
                n_heads,
                n_kv_heads,
                hidden_size,
                eps,
                attention_backend=attention_backend,
            )
            for _ in range(n_layers)
        ])
        self.norm = RmsNorm(dim, eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

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
                raise ValueError("TinyTransformer requires input_ids or hidden_states")
            if self.embed is None:
                raise ValueError("TinyTransformer PP non-first stage requires hidden_states input")
            x = self.embed(input_ids)
        b, s, d = x.size()
        if position_ids is None:
            cos, sin = self.rope(position_offset, position_offset + s)
        else:
            cos, sin = self.rope(position_ids=position_ids)
        for layer in self.layers:
            x = layer(
                x,
                cos,
                sin,
                position_offset=position_offset,
                position_ids=position_ids,
                sequence_ids=sequence_ids,
            )
        if self.norm is None or self.lm_head is None:
            return x
        x = self.norm(x)
        logits = self.lm_head(x)

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
            head_layers=["embed"],
            pipe_layers=["layers"],
            tail_layers=["norm", "lm_head"],
        )

    def context_parallel_spec(self) -> ContextParallelSpec:
        return ContextParallelSpec(
            attention_paths=[f"layers.{i}.attn" for i in range(len(self.layers))],
        )

class TinyTransformerTp(TinyTransformer):
    
    def tpsp_parallelize_spec(self):
        rules = []
        for i in range(len(self.layers)):
            rules += [
                TpSpShardRule(f"layers.{i}.attn.q_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.attn.k_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.attn.v_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.attn.o_proj", shard_axis=TpSpShardAxis.PARAM_IN, post_comm="all_reduce"),
                TpSpShardRule(f"layers.{i}.mlp.gate", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.mlp.up", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.mlp.down", shard_axis=TpSpShardAxis.PARAM_IN, post_comm="all_reduce"),
            ]
        return TpSpParallelSpec(
            rules=rules,
            tie_rules=[]
        )

class TinyTransformerTpSp(TinyTransformer):
    
    def tpsp_parallelize_spec(self):
        rules = [
            TpSpShardRule(f"embed", shard_axis=TpSpShardAxis.SEQUENCE, post_comm="scatter", comm_dim=1),
        ]
        for i in range(len(self.layers)):
            rules += [
                TpSpShardRule(f"layers.{i}.attn", shard_axis=TpSpShardAxis.SEQUENCE, pre_comm="all_gather", comm_dim=1),
                TpSpShardRule(f"layers.{i}.attn.q_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.attn.k_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.attn.v_proj", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.attn.o_proj", shard_axis=TpSpShardAxis.PARAM_IN, post_comm="reduce_scatter"),
                TpSpShardRule(f"layers.{i}.mlp", shard_axis=TpSpShardAxis.SEQUENCE, pre_comm="all_gather", comm_dim=1),
                TpSpShardRule(f"layers.{i}.mlp.gate", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.mlp.up", shard_axis=TpSpShardAxis.PARAM_OUT),
                TpSpShardRule(f"layers.{i}.mlp.down", shard_axis=TpSpShardAxis.PARAM_IN, post_comm="reduce_scatter"),
            ]
        rules += [
            TpSpShardRule(f"lm_head", shard_axis=TpSpShardAxis.SEQUENCE, pre_comm="all_gather", comm_dim=1),
        ]
        return TpSpParallelSpec(
            rules=rules,
            tie_rules=[]
        )
        
