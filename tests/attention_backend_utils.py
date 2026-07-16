from __future__ import annotations

import argparse

import torch

from runtime.layers.flash_utils import flash_attn_dense_block_fallback_reason, flash_attn_fallback_reason
from utils.attention_backend import AttentionBackend


ATTENTION_BACKEND_CHOICES = ("auto", AttentionBackend.EAGER, AttentionBackend.FLASH_ATTN)


def add_attention_backend_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--attention-backend",
        choices=ATTENTION_BACKEND_CHOICES,
        default="auto",
        help="Attention backend for test models. auto uses flash_attn only for CUDA/NCCL when available.",
    )


def resolve_attention_backend(
    requested: str,
    *,
    dist_backend: str | None = None,
    device: torch.device | None = None,
    require_dense_block: bool = False,
    allow_flash: bool = True,
) -> str:
    if requested != "auto":
        return requested
    if not allow_flash:
        return AttentionBackend.EAGER
    if dist_backend != "nccl":
        return AttentionBackend.EAGER
    if device is None or device.type != "cuda":
        return AttentionBackend.EAGER
    probe = torch.empty((1, 1, 1, 8), dtype=torch.float16, device=device)
    reason = flash_attn_dense_block_fallback_reason(probe) if require_dense_block else flash_attn_fallback_reason(probe)
    if reason is not None:
        return AttentionBackend.EAGER
    return AttentionBackend.FLASH_ATTN
