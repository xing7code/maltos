from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch.distributed as dist

from .schedule import PipelineScheduleConfig
from train_system.utils.logging import debug_log


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
        groups = self._build_groups()
        object.__setattr__(self, "groups", groups)

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

    def _build_groups(self) -> dict[MeshAxis, dist.ProcessGroup | None]:
        assert dist.is_initialized(), "dist must be initialized before build_groups!"
        groups = {
            MeshAxis.DP: None,
            MeshAxis.TP: None,
            MeshAxis.PP: None,
            MeshAxis.CP: None,
            MeshAxis.EP: None,
        }
        global_rank = dist.get_rank()

        ranks = np.arange(self.world_size).reshape(self.dp, self.pp, self.cp, self.tp)
        debug_log(3, f"mesh global ranks (shape=[{self.dp=}, {self.pp=}, {self.cp=}, {self.tp=}]): {ranks}")
        dp_indices, pp_indices, cp_indices, tp_indices = np.where(ranks==global_rank)
        assert len(dp_indices)==1, f"invalid rank {ranks}"
        dp_idx, pp_idx, cp_idx, tp_idx = dp_indices[0], pp_indices[0], cp_indices[0], tp_indices[0]

        # TP
        for d in range(self.dp):
            for p in range(self.pp):
                for c in range(self.cp):
                    group_ranks = ranks[d, p, c, :].tolist()
                    group = dist.new_group(group_ranks)
                    if d==dp_idx and p==pp_idx and c==cp_idx and self.tp>1:
                        debug_log(3, f"TP group global ranks: {group_ranks}")
                        groups[MeshAxis.TP] = group
        # CP
        for d in range(self.dp):
            for p in range(self.pp):
                for t in range(self.tp):
                    group_ranks = ranks[d, p, :, t].tolist()
                    group = dist.new_group(group_ranks)
                    if d==dp_idx and p==pp_idx and t==tp_idx and self.cp>1:
                        debug_log(3, f"CP group global ranks: {group_ranks}")
                        groups[MeshAxis.CP] = group
        # PP
        for d in range(self.dp):
            for c in range(self.cp):
                for t in range(self.tp):
                    group_ranks = ranks[d, :, c, t].tolist()
                    group = dist.new_group(group_ranks)
                    if d==dp_idx and c==cp_idx and t==tp_idx and self.pp>1:
                        debug_log(3, f"PP group global ranks: {group_ranks}")
                        groups[MeshAxis.PP] = group
        # DP
        for p in range(self.pp):
            for c in range(self.cp):
                for t in range(self.tp):
                    group_ranks = ranks[:, p, c, t].tolist()
                    group = dist.new_group(group_ranks)
                    if p==pp_idx and c==cp_idx and t==tp_idx and self.dp>1:
                        debug_log(3, f"DP group global ranks: {group_ranks}")
                        groups[MeshAxis.DP] = group
        return groups
        
    def get_group(self, axis: MeshAxis) -> dist.ProcessGroup | None:
        return self.groups[axis]


@dataclass(frozen=True)
class ParallelPlan:
    mesh: ProcessMesh
    zero_stage: int = 0  # 0, 1, 2, 3
    pp_schedule: PipelineScheduleConfig = field(default_factory=PipelineScheduleConfig)

    def __post_init__(self):
        if self.zero_stage not in (0, 1, 2, 3):
            raise ValueError(f"zero_stage must be 0/1/2/3, got {self.zero_stage}")
