from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .schedule import PipelineScheduleConfig


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

    def __post_init__(self):
        self._validate()

    @property
    def world_size(self) -> int:
        return self.dp * self.tp * self.pp * self.cp

    def _validate(self) -> None:
        for axis, size in {
            MeshAxis.DP: self.dp,
            MeshAxis.TP: self.tp,
            MeshAxis.PP: self.pp,
            MeshAxis.CP: self.cp,
            MeshAxis.EP: self.ep,
        }.items():
            if size < 1:
                raise ValueError(f"{axis.value} must be >= 1, got {size}")
        assert self.dp % self.ep == 0, f"EP must be a subset of DP, got dp={self.dp}, ep={self.ep}."


@dataclass(frozen=True)
class ParallelPlan:
    mesh: ProcessMesh
    zero_stage: int = 0  # 0, 1, 2, 3
    pp_schedule: PipelineScheduleConfig = field(default_factory=PipelineScheduleConfig)

    def __post_init__(self):
        if self.zero_stage not in (0, 1, 2, 3):
            raise ValueError(f"zero_stage must be 0/1/2/3, got {self.zero_stage}")
