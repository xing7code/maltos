from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class ParamShardMetadata:
    fq_name: str
    numel: int
    shard_offset: int
    shard_numel: int
    owner_rank: int


@dataclass
class ParamRuntimeMetadata:
    logical_shape: tuple[int, ...] = ()
    local_shape: tuple[int, ...] = ()
    is_sharded: bool = False
    is_materialized: bool = True
    materialized_by: str | None = None
    optimizer_visible: bool = True
    extra: dict[str, object] | None = None


@dataclass
class ParamHandle:
    """System-owned state handle for one logical parameter."""

    param: nn.Parameter
    shard: ParamShardMetadata
    runtime: ParamRuntimeMetadata
    is_gathered: bool = False

    def mark_gathered(self) -> None:
        self.is_gathered = True
        self.runtime.is_materialized = True

    def mark_sharded(self) -> None:
        self.is_gathered = False
        self.runtime.is_materialized = False
