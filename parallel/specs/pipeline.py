from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineParallelSpec:
    head_layers: list[str]
    pipe_layers: list[str]
    tail_layers: list[str]
