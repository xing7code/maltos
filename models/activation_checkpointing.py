from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActivationCheckpointConfig:
    enabled: bool = False
    granularity: str = "block"
    every_n_layers: int = 1

    def __post_init__(self) -> None:
        if self.granularity != "block":
            raise ValueError(f"only block-level activation checkpointing is supported, got {self.granularity!r}")
        if self.every_n_layers < 1:
            raise ValueError(f"every_n_layers must be >= 1, got {self.every_n_layers}")

    def should_checkpoint_layer(self, layer_idx: int) -> bool:
        return self.enabled and (layer_idx + 1) % self.every_n_layers == 0
