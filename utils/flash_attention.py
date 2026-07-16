from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.constants import PAD_SEQUENCE_ID

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    _FLASH_ATTN_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on optional package/runtime
    _flash_attn_func = None
    _flash_attn_varlen_func = None
    _FLASH_ATTN_IMPORT_ERROR = exc


@dataclass(frozen=True)
class VarLenPackedQkv:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    token_mask: torch.Tensor


@dataclass(frozen=True)
class VarLenPackedPrefixQkv:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    token_mask: torch.Tensor


def flash_attn_import_error() -> Exception | None:
    return _FLASH_ATTN_IMPORT_ERROR


def flash_attn_is_available() -> bool:
    return _flash_attn_func is not None and _flash_attn_varlen_func is not None


def flash_attn_fallback_reason(q: torch.Tensor) -> str | None:
    if not flash_attn_is_available():
        err = flash_attn_import_error()
        if err is None:
            return "flash-attn package is unavailable"
        return f"flash-attn package is unavailable: {type(err).__name__}: {err}"
    if not q.is_cuda:
        return "flash-attn requires CUDA tensors"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return f"flash-attn requires fp16/bf16 tensors, got {q.dtype}"
    return None


def flash_attn_dense(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if _flash_attn_func is None:
        raise RuntimeError("flash-attn is not available")
    out = _flash_attn_func(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        dropout_p=0.0,
        causal=True,
    )
    return out.transpose(1, 2).contiguous()


def pack_varlen_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sequence_ids: torch.Tensor,
) -> VarLenPackedQkv:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("pack_varlen_qkv expects q, k, v to have shape [batch, heads, seq, dim]")
    if sequence_ids.dim() != 2:
        raise ValueError(f"sequence_ids must have shape [batch, seq], got {tuple(sequence_ids.shape)}")
    batch_size, _, seq_len, _ = q.shape
    if sequence_ids.size(0) != batch_size or sequence_ids.size(1) != seq_len:
        raise ValueError(
            "sequence_ids shape must match q/k/v batch+seq dimensions, "
            f"got {tuple(sequence_ids.shape)} vs ({batch_size}, {seq_len})"
        )

    valid_tokens = sequence_ids != PAD_SEQUENCE_ID
    total_valid = int(valid_tokens.sum().item())
    if total_valid == 0:
        empty = q.new_empty((0, q.size(1), q.size(-1)))
        return VarLenPackedQkv(
            q=empty,
            k=empty,
            v=empty,
            cu_seqlens=torch.zeros(1, dtype=torch.int32, device=q.device),
            max_seqlen=0,
            token_mask=valid_tokens,
        )

    prev_valid = torch.zeros_like(valid_tokens)
    prev_valid[:, 1:] = valid_tokens[:, :-1]
    prev_sequence_ids = torch.zeros_like(sequence_ids)
    prev_sequence_ids[:, 1:] = sequence_ids[:, :-1]
    starts = valid_tokens & (~prev_valid | (sequence_ids != prev_sequence_ids))

    flat_valid = valid_tokens.reshape(-1)
    flat_starts = starts.reshape(-1)[flat_valid]
    segment_ids = flat_starts.to(dtype=torch.int32).cumsum(dim=0) - 1
    num_segments = int(segment_ids[-1].item()) + 1
    lengths = torch.bincount(segment_ids, minlength=num_segments)
    cu_seqlens = torch.zeros(num_segments + 1, dtype=torch.int32, device=q.device)
    cu_seqlens[1:] = lengths.cumsum(dim=0, dtype=torch.int32)
    max_seqlen = int(lengths.max().item())

    q_flat = q.transpose(1, 2).contiguous().reshape(batch_size * seq_len, q.size(1), q.size(-1))[flat_valid]
    k_flat = k.transpose(1, 2).contiguous().reshape(batch_size * seq_len, k.size(1), k.size(-1))[flat_valid]
    v_flat = v.transpose(1, 2).contiguous().reshape(batch_size * seq_len, v.size(1), v.size(-1))[flat_valid]
    return VarLenPackedQkv(
        q=q_flat,
        k=k_flat,
        v=v_flat,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        token_mask=valid_tokens,
    )


def unpack_varlen_output(
    out: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    batch_size, seq_len = token_mask.shape
    dense = out.new_zeros((batch_size * seq_len, num_heads, head_dim))
    dense[token_mask.reshape(-1)] = out
    return dense.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()


def flash_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sequence_ids: torch.Tensor,
) -> torch.Tensor:
    if _flash_attn_varlen_func is None:
        raise RuntimeError("flash-attn is not available")
    packed = pack_varlen_qkv(q, k, v, sequence_ids)
    if packed.max_seqlen == 0:
        return q.new_zeros(q.shape)
    out = _flash_attn_varlen_func(
        packed.q,
        packed.k,
        packed.v,
        packed.cu_seqlens,
        packed.cu_seqlens,
        packed.max_seqlen,
        packed.max_seqlen,
        dropout_p=0.0,
        causal=True,
    )
    return unpack_varlen_output(
        out,
        packed.token_mask,
        num_heads=q.size(1),
        head_dim=q.size(-1),
    )


def pack_varlen_prefix_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    q_sequence_ids: torch.Tensor | None = None,
    k_sequence_ids: torch.Tensor | None = None,
) -> VarLenPackedPrefixQkv:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("pack_varlen_prefix_qkv expects q, k, v to have shape [batch, heads, seq, dim]")
    if q_positions.dim() != 2 or k_positions.dim() != 2:
        raise ValueError("q_positions and k_positions must have shape [batch, seq]")
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    k_batch_size, k_num_heads, k_seq_len, k_head_dim = k.shape
    if (k_batch_size, k_num_heads, k_head_dim) != (batch_size, num_heads, head_dim):
        raise ValueError(
            "q and k/v must agree on batch/head/head_dim, "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if q_positions.shape != (batch_size, q_seq_len):
        raise ValueError(
            "q_positions shape must match q batch+seq dimensions, "
            f"got {tuple(q_positions.shape)} vs ({batch_size}, {q_seq_len})"
        )
    if k_positions.shape != (batch_size, k_seq_len):
        raise ValueError(
            "k_positions shape must match k/v batch+seq dimensions, "
            f"got {tuple(k_positions.shape)} vs ({batch_size}, {k_seq_len})"
        )

    if q_sequence_ids is None:
        q_sequence_ids = torch.zeros_like(q_positions)
    elif q_sequence_ids.shape != (batch_size, q_seq_len):
        raise ValueError(
            "q_sequence_ids shape must match q batch+seq dimensions, "
            f"got {tuple(q_sequence_ids.shape)} vs ({batch_size}, {q_seq_len})"
        )
    if k_sequence_ids is None:
        k_sequence_ids = torch.zeros_like(k_positions)
    elif k_sequence_ids.shape != (batch_size, k_seq_len):
        raise ValueError(
            "k_sequence_ids shape must match k/v batch+seq dimensions, "
            f"got {tuple(k_sequence_ids.shape)} vs ({batch_size}, {k_seq_len})"
        )

    q_valid = q_sequence_ids != PAD_SEQUENCE_ID
    k_valid = k_sequence_ids != PAD_SEQUENCE_ID
    q_flat = q.transpose(1, 2).contiguous().reshape(batch_size * q_seq_len, num_heads, head_dim)
    k_flat = k.transpose(1, 2).contiguous().reshape(batch_size * k_seq_len, num_heads, head_dim)
    v_flat = v.transpose(1, 2).contiguous().reshape(batch_size * k_seq_len, num_heads, head_dim)

    q_segments: list[torch.Tensor] = []
    k_segments: list[torch.Tensor] = []
    q_lengths: list[int] = []
    k_lengths: list[int] = []

    for batch_idx in range(batch_size):
        q_valid_idx = torch.nonzero(q_valid[batch_idx], as_tuple=False).flatten()
        if q_valid_idx.numel() == 0:
            continue
        run_start = 0
        while run_start < q_valid_idx.numel():
            run_end = run_start + 1
            run_indices = q_valid_idx[run_start:run_end]
            run_sequence_id = int(q_sequence_ids[batch_idx, run_indices[0]].item())
            prev_position = int(q_positions[batch_idx, run_indices[0]].item())
            while run_end < q_valid_idx.numel():
                next_idx = int(q_valid_idx[run_end].item())
                next_sequence_id = int(q_sequence_ids[batch_idx, next_idx].item())
                next_position = int(q_positions[batch_idx, next_idx].item())
                if next_sequence_id != run_sequence_id or next_position != prev_position + 1:
                    break
                prev_position = next_position
                run_end += 1
            run_indices = q_valid_idx[run_start:run_end]
            run_end_position = int(q_positions[batch_idx, run_indices[-1]].item())

            k_mask = (
                k_valid[batch_idx]
                & (k_sequence_ids[batch_idx] == run_sequence_id)
                & (k_positions[batch_idx] <= run_end_position)
            )
            k_indices = torch.nonzero(k_mask, as_tuple=False).flatten()
            if k_indices.numel() == 0:
                raise ValueError(
                    "pack_varlen_prefix_qkv found a query run without any matching key/value prefix; "
                    f"batch={batch_idx} sequence_id={run_sequence_id} end_position={run_end_position}"
                )

            q_segments.append(run_indices + batch_idx * q_seq_len)
            k_segments.append(k_indices + batch_idx * k_seq_len)
            q_lengths.append(int(run_indices.numel()))
            k_lengths.append(int(k_indices.numel()))
            run_start = run_end

    if not q_segments:
        empty = q.new_empty((0, num_heads, head_dim))
        return VarLenPackedPrefixQkv(
            q=empty,
            k=empty,
            v=empty,
            cu_seqlens_q=torch.zeros(1, dtype=torch.int32, device=q.device),
            cu_seqlens_k=torch.zeros(1, dtype=torch.int32, device=q.device),
            max_seqlen_q=0,
            max_seqlen_k=0,
            token_mask=q_valid,
        )

    q_indices = torch.cat(q_segments, dim=0)
    k_indices = torch.cat(k_segments, dim=0)
    q_lengths_tensor = torch.tensor(q_lengths, dtype=torch.int32, device=q.device)
    k_lengths_tensor = torch.tensor(k_lengths, dtype=torch.int32, device=q.device)
    cu_seqlens_q = torch.zeros(len(q_lengths) + 1, dtype=torch.int32, device=q.device)
    cu_seqlens_k = torch.zeros(len(k_lengths) + 1, dtype=torch.int32, device=q.device)
    cu_seqlens_q[1:] = q_lengths_tensor.cumsum(dim=0, dtype=torch.int32)
    cu_seqlens_k[1:] = k_lengths_tensor.cumsum(dim=0, dtype=torch.int32)
    return VarLenPackedPrefixQkv(
        q=q_flat.index_select(0, q_indices),
        k=k_flat.index_select(0, k_indices),
        v=v_flat.index_select(0, k_indices),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=int(q_lengths_tensor.max().item()),
        max_seqlen_k=int(k_lengths_tensor.max().item()),
        token_mask=q_valid,
    )


def flash_attn_varlen_prefix(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    q_sequence_ids: torch.Tensor | None = None,
    k_sequence_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if _flash_attn_varlen_func is None:
        raise RuntimeError("flash-attn is not available")
    packed = pack_varlen_prefix_qkv(
        q,
        k,
        v,
        q_positions=q_positions,
        k_positions=k_positions,
        q_sequence_ids=q_sequence_ids,
        k_sequence_ids=k_sequence_ids,
    )
    if packed.max_seqlen_q == 0 or packed.max_seqlen_k == 0:
        return q.new_zeros(q.shape)
    out = _flash_attn_varlen_func(
        packed.q,
        packed.k,
        packed.v,
        packed.cu_seqlens_q,
        packed.cu_seqlens_k,
        packed.max_seqlen_q,
        packed.max_seqlen_k,
        dropout_p=0.0,
        causal=True,
    )
    return unpack_varlen_output(
        out,
        packed.token_mask,
        num_heads=q.size(1),
        head_dim=q.size(-1),
    )
