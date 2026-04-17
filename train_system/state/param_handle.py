from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParamShardMetadata:
    fq_name: str
    numel: int
    shard_offset: int
    shard_numel: int
    owner_rank: int


@dataclass
class ParamHandle:
    """Metadata wrapper around a regular torch Parameter.

    Keep torch.nn.Parameter untouched for ecosystem compatibility.
    """

    shard: ParamShardMetadata
    is_gathered: bool = False

    def mark_gathered(self) -> None:
        self.is_gathered = True

    def mark_sharded(self) -> None:
        self.is_gathered = False
