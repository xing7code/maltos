from dataclasses import dataclass
from enum import Enum


class PipelineScheduleType(str, Enum):
    ONE_FWD_ONE_BWD = "1f1b"
    INTERLEAVED_1F1B = "interleaved_1f1b"
    ZERO_BUBBLE = "zero_bubble"


@dataclass(frozen=True)
class PipelineScheduleConfig:
    schedule: PipelineScheduleType = PipelineScheduleType.ONE_FWD_ONE_BWD
    virtual_stages: int = 1
    microbatches: int = 1

    def __post_init__(self) -> None:
        if self.virtual_stages < 1:
            raise ValueError(f"virtual_stages must be >= 1, got {self.virtual_stages}")
        if self.microbatches < 1:
            raise ValueError(f"microbatches must be >= 1, got {self.microbatches}")
