from __future__ import annotations

from dataclasses import dataclass, field

from .mesh import ProcessMesh
from .schedule import PipelineScheduleConfig


@dataclass(frozen=True)
class ParallelConfig:
    use_ddp: bool = True
    ddp_bucket_cap_mb: int = 25
    ddp_async_allreduce: bool = True

    use_tp: bool = False
    use_sp: bool = False

    use_pp: bool = False
    pp: PipelineScheduleConfig = field(default_factory=PipelineScheduleConfig)

    use_cp: bool = False
    use_ep: bool = False

    zero_stage: int = 0  # 0, 1, 2, 3


@dataclass(frozen=True)
class ParallelPlan:
    mesh: ProcessMesh
    config: ParallelConfig

    def validate(self) -> None:
        self.mesh.validate()

        if self.config.zero_stage not in (0, 1, 2, 3):
            raise ValueError(f"zero_stage must be 0/1/2/3, got {self.config.zero_stage}")

        if self.config.use_pp and self.mesh.pp <= 1:
            raise ValueError("use_pp=True requires mesh.pp > 1")

        if self.config.use_tp and self.mesh.tp <= 1:
            raise ValueError("use_tp=True requires mesh.tp > 1")

        if self.config.use_cp and self.mesh.cp <= 1:
            raise ValueError("use_cp=True requires mesh.cp > 1")

        if self.config.use_ep and self.mesh.ep <= 1:
            raise ValueError("use_ep=True requires mesh.ep > 1")
