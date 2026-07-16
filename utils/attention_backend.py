from __future__ import annotations

from enum import StrEnum
from typing import Literal

import torch
import torch.nn.functional as F
from absl import logging
from torch.nn.attention import SDPBackend, sdpa_kernel

from utils.attention_masking import build_example_causal_mask
from utils.flash_attention import (
    flash_attn_dense,
    flash_attn_fallback_reason,
    flash_attn_varlen,
    flash_attn_varlen_prefix,
)


class AttentionBackend(StrEnum):
    EAGER = "eager"
    SDPA_AUTO = "sdpa_auto"
    SDPA_FLASH = "sdpa_flash"
    FLASH_ATTN = "flash_attn"


FlashVarlenMode = Literal["same_length", "prefix"]
ATTENTION_BACKEND_CHOICES: tuple[str, ...] = tuple(backend.value for backend in AttentionBackend)

_WARNED_KEYS: set[str] = set()


def validate_attention_backend(attention_backend: str | AttentionBackend) -> AttentionBackend:
    try:
        return AttentionBackend(attention_backend)
    except ValueError:
        raise ValueError(
            f"attention_backend must be one of {sorted(ATTENTION_BACKEND_CHOICES)}, got {attention_backend!r}"
        )


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attention_backend: str,
    warning_prefix: str,
    q_positions: torch.Tensor | None = None,
    k_positions: torch.Tensor | None = None,
    q_sequence_ids: torch.Tensor | None = None,
    k_sequence_ids: torch.Tensor | None = None,
    flash_varlen_mode: FlashVarlenMode | None = None,
) -> torch.Tensor:
    has_example_layout = any(
        value is not None
        for value in (q_positions, k_positions, q_sequence_ids, k_sequence_ids)
    )
    if not has_example_layout:
        if attention_backend == AttentionBackend.SDPA_AUTO:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if attention_backend == AttentionBackend.SDPA_FLASH:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if attention_backend == AttentionBackend.FLASH_ATTN:
            fallback_reason = flash_attn_fallback_reason(q)
            if fallback_reason is None:
                return flash_attn_dense(q, k, v)
            _warn_once(
                f"{warning_prefix}.flash_attn_fallback",
                "%s flash_attn backend falling back to regular attention: %s",
                warning_prefix,
                fallback_reason,
            )
        return eager_causal_attention(q, k, v)

    if q_positions is None or k_positions is None:
        raise ValueError("q_positions and k_positions must both be provided when using example-aware attention")

    mask = build_example_causal_mask(
        q_positions=q_positions,
        k_positions=k_positions,
        q_sequence_ids=q_sequence_ids,
        k_sequence_ids=k_sequence_ids,
    ).unsqueeze(1)
    if attention_backend == AttentionBackend.SDPA_AUTO:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    if attention_backend == AttentionBackend.SDPA_FLASH:
        _warn_once(
            f"{warning_prefix}.sdpa_flash_mask_fallback",
            "%s sdpa_flash received example-aware attention mask; falling back to regular SDPA for correctness",
            warning_prefix,
        )
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    if attention_backend == AttentionBackend.FLASH_ATTN:
        fallback_reason = flash_attn_fallback_reason(q)
        if fallback_reason is None:
            if flash_varlen_mode == "same_length" and q_sequence_ids is not None:
                return flash_attn_varlen(q, k, v, q_sequence_ids)
            if flash_varlen_mode == "prefix":
                return flash_attn_varlen_prefix(
                    q,
                    k,
                    v,
                    q_positions=q_positions,
                    k_positions=k_positions,
                    q_sequence_ids=q_sequence_ids,
                    k_sequence_ids=k_sequence_ids,
                )
            fallback_reason = "flash-attn requires compatible sequence metadata for this example-aware layout"
        _warn_once(
            f"{warning_prefix}.flash_attn_fallback",
            "%s flash_attn backend falling back to regular attention: %s",
            warning_prefix,
            fallback_reason,
        )
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
    return eager_causal_attention(q, k, v, mask=mask)


def eager_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if mask is None:
        seq_len = q.size(-2)
        mask = torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool).tril().unsqueeze(0).unsqueeze(0)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def _warn_once(key: str, msg: str, *args: object) -> None:
    if key in _WARNED_KEYS:
        return
    logging.warning(msg, *args)
    _WARNED_KEYS.add(key)
