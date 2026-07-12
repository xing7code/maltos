from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextParallelSpec:
    attention_paths: list[str]
