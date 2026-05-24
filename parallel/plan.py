from __future__ import annotations

from dataclasses import dataclass, field

from .schedule import PipelineScheduleConfig


@dataclass(frozen=True)
class ParallelPlan:
    zero_stage: int = 0  # 0, 1, 2, 3
    pp_schedule: PipelineScheduleConfig = field(default_factory=PipelineScheduleConfig)

    def __post_init__(self):
        if self.zero_stage not in (0, 1, 2, 3):
            raise ValueError(f"zero_stage must be 0/1/2/3, got {self.zero_stage}")
