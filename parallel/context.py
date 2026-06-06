from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class ContextParallelSpec:
    attention_paths: list[str]


class ContextParallelAttentionCoreType(str, Enum):
    ALL_GATHER_KV = "all_gather_kv"
    RING = "ring"


@runtime_checkable
class ContextParallelAttentionCore(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_offset: int,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
