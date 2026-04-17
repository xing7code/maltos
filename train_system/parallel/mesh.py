from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MeshAxis(str, Enum):
    DP = "dp"
    TP = "tp"
    PP = "pp"
    CP = "cp"
    EP = "ep"


@dataclass(frozen=True)
class ProcessMesh:
    dp: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1

    @property
    def world_size(self) -> int:
        return self.dp * self.tp * self.pp * self.cp * self.ep

    def validate(self) -> None:
        for axis, size in {
            MeshAxis.DP: self.dp,
            MeshAxis.TP: self.tp,
            MeshAxis.PP: self.pp,
            MeshAxis.CP: self.cp,
            MeshAxis.EP: self.ep,
        }.items():
            if size < 1:
                raise ValueError(f"{axis.value} must be >= 1, got {size}")
