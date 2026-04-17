from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CheckpointSpec:
    format: str = "sharded"
    include_optimizer: bool = True
    include_rng_state: bool = True
