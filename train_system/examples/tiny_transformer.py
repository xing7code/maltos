from __future__ import annotations

from absl import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from train_system.parallel.spec import OpShardRule, ModelParallelSpec


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.head_dim = dim//n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2*self.head_dim*self.n_kv_heads, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos=None, sin=None):
        if dist.get_rank()==0:
            logging.vlog(3, f"layer: {self._get_name()} input shape={x.size()}")
        b, s, d = x.size()
        # use -1 for n_heads, n_kv_heads, this dim can be sharded for TP.
        q = self.q_proj(x).view(b, s, -1, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(x).view(b, s, -1).chunk(2, dim=-1)
        k = k.view(b, s, -1, self.head_dim).transpose(1,2)
        v = v.view(b, s, -1, self.head_dim).transpose(1,2)
        if self.n_heads != self.n_kv_heads:
            k = k.repeat_interleave(self.n_heads//self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads//self.n_kv_heads, dim=1)
        logits = (q @ k.transpose(-1,-2)) / self.head_dim ** 0.5
        mask = torch.ones(s, s, device=x.device).triu(1).bool()
        logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0), torch.finfo(logits.dtype).min)
        scores = F.softmax(logits, dim=-1)
        # use -1 for last dim, it can be different than d for TP.
        out = (scores @ v).transpose(1, 2).contiguous().view(b, s, -1)
        output = self.o_proj(out)
        if dist.get_rank()==0:
            logging.vlog(3, f"layer: {self._get_name()}, output shape={output.size()}")
        return output


class MLP(nn.Module):
    def __init__(self, dim, hidden_size):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_size, bias=False)
        self.up = nn.Linear(dim, hidden_size, bias=False)
        self.down = nn.Linear(hidden_size, dim, bias=False)
    
    def forward(self, x):
        if dist.get_rank()==0:
            logging.vlog(3, f"layer: {self._get_name()}, input shape={x.size()}")
        output = self.down(F.silu(self.gate(x)) * self.up(x))
        if dist.get_rank()==0:
            logging.vlog(3, f"layer: {self._get_name()}, output shape={output.size()}")
        return output


class RmsNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if dist.get_rank()==0:
            logging.vlog(3, f"layer: {self._get_name()}, input shape={x.size()}")
        output = self.weight * x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        if dist.get_rank()==0:
            logging.vlog(3, f"layer: {self._get_name()}, output shape={output.size()}")
        return output


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, hidden_size, eps):
        super().__init__()
        self.attn = CausalSelfAttention(dim, n_heads, n_kv_heads)
        self.norm1 = RmsNorm(dim, eps)
        self.mlp = MLP(dim, hidden_size)
        self.norm2 = RmsNorm(dim, eps)
    
    def forward(self, x, cos=None, sin=None):
        x = self.attn(self.norm1(x), cos, sin) + x
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

    def forward(self, start, end):
        return self.cos[start:end].unsqueeze(0).unsqueeze(0), self.sin[start:end].unsqueeze(0).unsqueeze(0)


class TinyTransformer(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, hidden_size, eps, n_layers, vocab_size, max_seq_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.rope = RoPE(dim//n_heads, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, hidden_size, eps) for _ in range(n_layers)
        ])
        self.norm = RmsNorm(dim, eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, batch):
        if isinstance(batch, (tuple, list)):
            input_ids, labels = batch
        else:
            input_ids, labels = batch, None

        x = self.embed(input_ids)
        b, s, d = x.size()
        cos, sin = self.rope(0, s)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)

        if labels is None:
            return logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss


class TinyTransformerTp(TinyTransformer):
    
    def parallelize_spec(self):
        rules = []
        for i in range(len(self.layers)):
            rules += [
                OpShardRule(f"layers.{i}.attn.q_proj", shard_style="col", shard_axis="out"),
                OpShardRule(f"layers.{i}.attn.kv_proj", shard_style="col", shard_axis="out"),
                OpShardRule(f"layers.{i}.attn.o_proj", shard_style="row", shard_axis="in", post_comm="all_reduce"),
                OpShardRule(f"layers.{i}.mlp.gate", shard_style="col", shard_axis="out"),
                OpShardRule(f"layers.{i}.mlp.up", shard_style="col", shard_axis="out"),
                OpShardRule(f"layers.{i}.mlp.down", shard_style="row", shard_axis="in", post_comm="all_reduce"),
            ]
        return ModelParallelSpec(
            rules=rules,
            tie_rules=[]
        )

class TinyTransformerTpSp(TinyTransformer):
    
    def parallelize_spec(self):
        rules = [
            OpShardRule(f"embed", shard_style="seq", post_comm="scatter", comm_dim=1),
        ]
        for i in range(len(self.layers)):
            rules += [
                OpShardRule(f"layers.{i}.attn", shard_style="seq", pre_comm="all_gather", comm_dim=1),
                OpShardRule(f"layers.{i}.attn.q_proj", shard_style="col", shard_axis="out"),
                OpShardRule(f"layers.{i}.attn.kv_proj", shard_style="col", shard_axis="out"),
                OpShardRule(f"layers.{i}.attn.o_proj", shard_style="row", shard_axis="in", post_comm="reduce_scatter"),
                OpShardRule(f"layers.{i}.mlp", shard_style="seq", pre_comm="all_gather", comm_dim=1),
                OpShardRule(f"layers.{i}.mlp.gate", shard_style="col", shard_axis="out"),
                OpShardRule(f"layers.{i}.mlp.up", shard_style="col", shard_axis="out"),
                OpShardRule(f"layers.{i}.mlp.down", shard_style="row", shard_axis="in", post_comm="reduce_scatter"),
            ]
        rules += [
            OpShardRule(f"lm_head", shard_style="seq", pre_comm="all_gather", comm_dim=1),
        ]
        return ModelParallelSpec(
            rules=rules,
            tie_rules=[]
        )
        