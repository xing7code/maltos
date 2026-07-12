from dataclasses import dataclass
from enum import Enum


class PipelineScheduleType(str, Enum):
    ONE_FWD_ONE_BWD = "1f1b"


@dataclass(frozen=True)
class PipelineScheduleConfig:
    schedule: PipelineScheduleType = PipelineScheduleType.ONE_FWD_ONE_BWD
    microbatches: int = 1

    def __post_init__(self) -> None:
        if self.microbatches < 1:
            raise ValueError(f"microbatches must be >= 1, got {self.microbatches}")
