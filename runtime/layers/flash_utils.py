from __future__ import annotations

from dataclasses import dataclass
import inspect

import torch

from utils.constants import PAD_SEQUENCE_ID

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    from flash_attn.flash_attn_interface import _flash_attn_backward as _flash_attn_backward_with_lse
    from flash_attn.flash_attn_interface import _flash_attn_forward as _flash_attn_forward_with_lse
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward as _flash_attn_varlen_backward_with_lse
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward as _flash_attn_varlen_forward_with_lse
    _FLASH_ATTN_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on optional package/runtime
    _flash_attn_func = None
    _flash_attn_varlen_func = None
    _flash_attn_backward_with_lse = None
    _flash_attn_forward_with_lse = None
    _flash_attn_varlen_backward_with_lse = None
    _flash_attn_varlen_forward_with_lse = None
    _FLASH_ATTN_IMPORT_ERROR = exc


@dataclass(frozen=True)
class PackedQkv:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    token_mask: torch.Tensor


@dataclass(frozen=True)
class PackedPrefixQkv:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_indices: torch.Tensor
    k_indices: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    token_mask: torch.Tensor


@dataclass(frozen=True)
class FlashAttnBlockOutput:
    out: torch.Tensor
    lse: torch.Tensor


def flash_attn_import_error() -> Exception | None:
    return _FLASH_ATTN_IMPORT_ERROR


def flash_attn_is_available() -> bool:
    return _flash_attn_func is not None and _flash_attn_varlen_func is not None


def flash_attn_block_is_available() -> bool:
    return (
        flash_attn_is_available()
        and _flash_attn_forward_with_lse is not None
        and _flash_attn_backward_with_lse is not None
        and _flash_attn_varlen_forward_with_lse is not None
        and _flash_attn_varlen_backward_with_lse is not None
    )


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


def flash_attn_block_fallback_reason(q: torch.Tensor) -> str | None:
    reason = flash_attn_fallback_reason(q)
    if reason is not None:
        return reason
    if _flash_attn_forward_with_lse is None or _flash_attn_backward_with_lse is None:
        return "flash-attn internal dense forward/backward helpers are unavailable"
    if _flash_attn_varlen_forward_with_lse is None or _flash_attn_varlen_backward_with_lse is None:
        return "flash-attn internal varlen forward/backward helpers are unavailable"
    return None


def flash_attn_dense_block_fallback_reason(q: torch.Tensor) -> str | None:
    reason = flash_attn_fallback_reason(q)
    if reason is not None:
        return reason
    if _flash_attn_forward_with_lse is None or _flash_attn_backward_with_lse is None:
        return "flash-attn internal dense forward/backward helpers are unavailable"
    return None


def _call_flash_attn_interface(func, **kwargs):
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)
    supported_kwargs = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return func(**supported_kwargs)


def _normalize_dense_softmax_lse(
    softmax_lse: torch.Tensor,
    *,
    batch_size: int,
    num_heads: int,
    seq_len: int,
) -> torch.Tensor:
    if softmax_lse.dim() == 4 and softmax_lse.size(-1) == 1:
        softmax_lse = softmax_lse.squeeze(dim=-1)
    if softmax_lse.shape == (batch_size, num_heads, seq_len):
        return softmax_lse.contiguous()
    if softmax_lse.shape == (batch_size, seq_len, num_heads):
        return softmax_lse.transpose(1, 2).contiguous()
    raise RuntimeError(
        "unexpected flash-attn dense softmax_lse shape, "
        f"got {tuple(softmax_lse.shape)} expected ({batch_size}, {num_heads}, {seq_len})"
    )


def _dense_to_flash_layout(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def _flash_to_dense_layout(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).contiguous()


def flash_attn_dense_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> FlashAttnBlockOutput:
    if _flash_attn_forward_with_lse is None:
        raise RuntimeError("flash-attn internal dense forward with softmax_lse is unavailable")
    outputs = _call_flash_attn_interface(
        _flash_attn_forward_with_lse,
        q=_dense_to_flash_layout(q),
        k=_dense_to_flash_layout(k),
        v=_dense_to_flash_layout(v),
        dropout_p=0.0,
        softmax_scale=q.size(-1) ** -0.5,
        causal=causal,
        return_softmax=False,
        window_size_left=-1,
        window_size_right=-1,
        alibi_slopes=None,
    )
    if len(outputs) == 8:
        out, _, _, _, _, softmax_lse, _, _ = outputs
    else:
        out, softmax_lse, _, _ = outputs
    return FlashAttnBlockOutput(
        out=_flash_to_dense_layout(out),
        lse=_normalize_dense_softmax_lse(
            softmax_lse,
            batch_size=q.size(0),
            num_heads=q.size(1),
            seq_len=q.size(2),
        ),
    )


def flash_attn_dense_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    *,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _flash_attn_backward_with_lse is None:
        raise RuntimeError("flash-attn internal dense backward helper is unavailable")
    flash_q = _dense_to_flash_layout(q)
    flash_k = _dense_to_flash_layout(k)
    flash_v = _dense_to_flash_layout(v)
    flash_dout = _dense_to_flash_layout(dout)
    flash_out = _dense_to_flash_layout(out)
    flash_softmax_lse = softmax_lse.contiguous()
    flash_dq = torch.empty_like(flash_q)
    flash_dk = torch.empty_like(flash_k)
    flash_dv = torch.empty_like(flash_v)
    _call_flash_attn_interface(
        _flash_attn_backward_with_lse,
        dout=flash_dout,
        q=flash_q,
        k=flash_k,
        v=flash_v,
        out=flash_out,
        softmax_lse=flash_softmax_lse,
        dq=flash_dq,
        dk=flash_dk,
        dv=flash_dv,
        dropout_p=0.0,
        softmax_scale=q.size(-1) ** -0.5,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
    )
    return (
        _flash_to_dense_layout(flash_dq),
        _flash_to_dense_layout(flash_dk),
        _flash_to_dense_layout(flash_dv),
    )


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


def pack_dense_varlen_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> PackedQkv:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("pack_dense_varlen_qkv expects q, k, v to have shape [batch, heads, seq, dim]")
    batch_size, num_heads, seq_len, head_dim = q.shape
    expected_shape = (batch_size, num_heads, seq_len, head_dim)
    if k.shape != expected_shape or v.shape != expected_shape:
        raise ValueError(
            "q, k, v must have identical shapes for dense varlen packing, "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    flat_q = q.transpose(1, 2).contiguous().view(batch_size * seq_len, num_heads, head_dim)
    flat_k = k.transpose(1, 2).contiguous().view(batch_size * seq_len, num_heads, head_dim)
    flat_v = v.transpose(1, 2).contiguous().view(batch_size * seq_len, num_heads, head_dim)
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seq_len,
        seq_len,
        dtype=torch.int32,
        device=q.device,
    )
    return PackedQkv(
        q=flat_q,
        k=flat_k,
        v=flat_v,
        cu_seqlens=cu_seqlens,
        max_seqlen=seq_len,
        token_mask=torch.ones((batch_size, seq_len), dtype=torch.bool, device=q.device),
    )


def pack_varlen_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sequence_ids: torch.Tensor,
) -> PackedQkv:
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
        return PackedQkv(
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
    return PackedQkv(
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


def unpack_varlen_output_by_indices(
    out: torch.Tensor,
    q_indices: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    dense = out.new_zeros((batch_size * seq_len, num_heads, head_dim))
    if q_indices.numel() > 0:
        dense[q_indices] = out
    return dense.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()


def gather_varlen_output_by_indices(
    dense: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    if dense.dim() != 4:
        raise ValueError(f"gather_varlen_output_by_indices expects [batch, heads, seq, dim], got {tuple(dense.shape)}")
    batch_size, num_heads, seq_len, head_dim = dense.shape
    flat = dense.transpose(1, 2).contiguous().reshape(batch_size * seq_len, num_heads, head_dim)
    return flat.index_select(0, indices)


def unpack_varlen_lse_by_indices(
    softmax_lse: torch.Tensor,
    q_indices: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    device: torch.device,
) -> torch.Tensor:
    dense = torch.full(
        (batch_size * seq_len, num_heads),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    if q_indices.numel() > 0:
        dense[q_indices] = softmax_lse.transpose(0, 1)
    return dense.view(batch_size, seq_len, num_heads).permute(0, 2, 1).contiguous()


def gather_varlen_lse_by_indices(
    dense_lse: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    if dense_lse.dim() != 3:
        raise ValueError(f"gather_varlen_lse_by_indices expects [batch, heads, seq], got {tuple(dense_lse.shape)}")
    batch_size, num_heads, seq_len = dense_lse.shape
    flat = dense_lse.permute(0, 2, 1).contiguous().reshape(batch_size * seq_len, num_heads)
    return flat.index_select(0, indices).transpose(0, 1).contiguous()


def scatter_varlen_output_by_indices(
    packed: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    dense = packed.new_zeros((batch_size * seq_len, num_heads, head_dim))
    if indices.numel() > 0:
        dense.index_add_(0, indices, packed)
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
    allow_empty_kv: bool = False,
) -> PackedPrefixQkv:
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
                if allow_empty_kv:
                    run_start = run_end
                    continue
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
        return PackedPrefixQkv(
            q=empty,
            k=empty,
            v=empty,
            q_indices=torch.zeros(0, dtype=torch.long, device=q.device),
            k_indices=torch.zeros(0, dtype=torch.long, device=q.device),
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
    return PackedPrefixQkv(
        q=q_flat.index_select(0, q_indices),
        k=k_flat.index_select(0, k_indices),
        v=v_flat.index_select(0, k_indices),
        q_indices=q_indices,
        k_indices=k_indices,
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
    return unpack_varlen_output_by_indices(
        out,
        packed.q_indices,
        batch_size=q.size(0),
        seq_len=q.size(2),
        num_heads=q.size(1),
        head_dim=q.size(-1),
    )


def flash_attn_varlen_prefix_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    q_sequence_ids: torch.Tensor | None = None,
    k_sequence_ids: torch.Tensor | None = None,
    allow_empty_kv: bool = False,
) -> FlashAttnBlockOutput:
    if _flash_attn_varlen_forward_with_lse is None:
        raise RuntimeError("flash-attn internal varlen forward with softmax_lse is unavailable")
    packed = pack_varlen_prefix_qkv(
        q,
        k,
        v,
        q_positions=q_positions,
        k_positions=k_positions,
        q_sequence_ids=q_sequence_ids,
        k_sequence_ids=k_sequence_ids,
        allow_empty_kv=allow_empty_kv,
    )
    batch_size, num_heads, seq_len, head_dim = q.shape
    if packed.max_seqlen_q == 0 or packed.max_seqlen_k == 0 or packed.q.numel() == 0:
        return FlashAttnBlockOutput(
            out=q.new_zeros(q.shape),
            lse=torch.full(
                (batch_size, num_heads, seq_len),
                float("-inf"),
                dtype=torch.float32,
                device=q.device,
            ),
        )
    outputs = _flash_attn_varlen_forward_with_lse(
        packed.q,
        packed.k,
        packed.v,
        packed.cu_seqlens_q,
        packed.cu_seqlens_k,
        packed.max_seqlen_q,
        packed.max_seqlen_k,
        0.0,
        q.size(-1) ** -0.5,
        True,
        return_softmax=False,
    )
    if len(outputs) == 8:
        out, _, _, _, _, softmax_lse, _, _ = outputs
    else:
        out, softmax_lse, _, _ = outputs
    return FlashAttnBlockOutput(
        out=unpack_varlen_output_by_indices(
            out,
            packed.q_indices,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        ),
        lse=unpack_varlen_lse_by_indices(
            softmax_lse,
            packed.q_indices,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            device=q.device,
        ),
    )


def flash_attn_varlen_prefix_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    *,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    q_sequence_ids: torch.Tensor | None = None,
    k_sequence_ids: torch.Tensor | None = None,
    allow_empty_kv: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _flash_attn_varlen_backward_with_lse is None:
        raise RuntimeError("flash-attn internal varlen backward helper is unavailable")
    packed = pack_varlen_prefix_qkv(
        q,
        k,
        v,
        q_positions=q_positions,
        k_positions=k_positions,
        q_sequence_ids=q_sequence_ids,
        k_sequence_ids=k_sequence_ids,
        allow_empty_kv=allow_empty_kv,
    )
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    k_seq_len = k.size(2)
    if packed.max_seqlen_q == 0 or packed.max_seqlen_k == 0 or packed.q.numel() == 0:
        return (
            q.new_zeros(q.shape),
            k.new_zeros(k.shape),
            v.new_zeros(v.shape),
        )

    packed_dout = gather_varlen_output_by_indices(dout, packed.q_indices)
    packed_out = gather_varlen_output_by_indices(out, packed.q_indices)
    packed_lse = gather_varlen_lse_by_indices(softmax_lse, packed.q_indices)
    packed_dq = torch.empty_like(packed.q)
    packed_dk = torch.empty_like(packed.k)
    packed_dv = torch.empty_like(packed.v)
    _flash_attn_varlen_backward_with_lse(
        dout=packed_dout,
        q=packed.q,
        k=packed.k,
        v=packed.v,
        out=packed_out,
        softmax_lse=packed_lse,
        dq=packed_dq,
        dk=packed_dk,
        dv=packed_dv,
        cu_seqlens_q=packed.cu_seqlens_q,
        cu_seqlens_k=packed.cu_seqlens_k,
        max_seqlen_q=packed.max_seqlen_q,
        max_seqlen_k=packed.max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=q.size(-1) ** -0.5,
        causal=True,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
    )
    return (
        scatter_varlen_output_by_indices(
            packed_dq,
            packed.q_indices,
            batch_size=batch_size,
            seq_len=q_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        ),
        scatter_varlen_output_by_indices(
            packed_dk,
            packed.k_indices,
            batch_size=batch_size,
            seq_len=k_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        ),
        scatter_varlen_output_by_indices(
            packed_dv,
            packed.k_indices,
            batch_size=batch_size,
            seq_len=k_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        ),
    )
