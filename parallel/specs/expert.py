from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertParallelSpec:
    moe_paths: list[str]
