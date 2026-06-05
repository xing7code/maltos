from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch.distributed as dist

from parallel.plan import ParallelPlan


class MeshAxis(str, Enum):
    DP = "dp"
    TP = "tp"
    PP = "pp"
    CP = "cp"
    EP = "ep"
    EDP = "edp"


@dataclass(frozen=True)
class MeshConfig:
    """Pure topology config for runtime process-group construction."""

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

    def rank_coordinates(self, rank_id: int) -> tuple[int, int, int, int]:
        if rank_id < 0 or rank_id >= self.world_size:
            raise ValueError(f"rank_id must be in [0, {self.world_size}), got {rank_id}")
        tp_idx = rank_id % self.tp
        rank_id //= self.tp
        cp_idx = rank_id % self.cp
        rank_id //= self.cp
        pp_idx = rank_id % self.pp
        rank_id //= self.pp
        dp_idx = rank_id
        return dp_idx, pp_idx, cp_idx, tp_idx

    def rank_id(self, *, dp: int, pp: int, cp: int, tp: int) -> int:
        if not (0 <= dp < self.dp and 0 <= pp < self.pp and 0 <= cp < self.cp and 0 <= tp < self.tp):
            raise ValueError(f"invalid mesh coordinates: dp={dp}, pp={pp}, cp={cp}, tp={tp}")
        return (((dp * self.pp) + pp) * self.cp + cp) * self.tp + tp

    def _validate(self) -> None:
        for axis, size in {
            MeshAxis.DP: self.dp,
            MeshAxis.TP: self.tp,
            MeshAxis.PP: self.pp,
            MeshAxis.CP: self.cp,
            MeshAxis.EP: self.ep,
            MeshAxis.EDP: self.dp // self.ep,
        }.items():
            if size < 1:
                raise ValueError(f"{axis.value} must be >= 1, got {size}")
        assert self.dp % self.ep == 0, f"EP must be a subset of DP, got dp={self.dp}, ep={self.ep}."


@dataclass
class ProcessGroupManager:
    mesh: MeshConfig
    groups: dict[MeshAxis, dist.ProcessGroup | None] = field(default_factory=dict)

    @classmethod
    def from_mesh(cls, mesh: MeshConfig) -> "ProcessGroupManager":
        manager = cls(mesh=mesh)
        manager.initialize_groups()
        return manager

    @classmethod
    def from_plan(cls, plan: ParallelPlan, mesh: MeshConfig | None = None) -> "ProcessGroupManager":
        if mesh is None:
            raise ValueError("from_plan requires mesh because ParallelPlan does not own MeshConfig")
        return cls.from_mesh(mesh)

    @property
    def world_size(self) -> int:
        return self.mesh.world_size

    def initialize_groups(self) -> None:
        self.groups = {
            MeshAxis.DP: None,
            MeshAxis.TP: None,
            MeshAxis.PP: None,
            MeshAxis.CP: None,
            MeshAxis.EP: None,
            MeshAxis.EDP: None,
        }
        if not dist.is_initialized():
            return

        global_rank = dist.get_rank()
        ranks = np.arange(self.world_size).reshape(self.mesh.dp, self.mesh.pp, self.mesh.cp, self.mesh.tp)
        dp_indices, pp_indices, cp_indices, tp_indices = np.where(ranks == global_rank)
        if len(dp_indices) != 1:
            raise ValueError(f"unable to locate global_rank={global_rank} in mesh ranks={ranks.tolist()}")
        dp_idx, pp_idx, cp_idx, tp_idx = dp_indices[0], pp_indices[0], cp_indices[0], tp_indices[0]

        for d in range(self.mesh.dp):
            for p in range(self.mesh.pp):
                for c in range(self.mesh.cp):
                    group_ranks = ranks[d, p, c, :].tolist()
                    group = dist.new_group(group_ranks)
                    if d == dp_idx and p == pp_idx and c == cp_idx and self.mesh.tp > 1:
                        self.groups[MeshAxis.TP] = group
        if self.mesh.ep > 1:
            edp = self.mesh.dp // self.mesh.ep
            for e in range(edp):
                dp_slice = slice(e * self.mesh.ep, (e + 1) * self.mesh.ep)
                for p in range(self.mesh.pp):
                    for c in range(self.mesh.cp):
                        for t in range(self.mesh.tp):
                            group_ranks = ranks[dp_slice, p, c, t].tolist()
                            group = dist.new_group(group_ranks)
                            if (
                                e == (dp_idx // self.mesh.ep)
                                and p == pp_idx
                                and c == cp_idx
                                and t == tp_idx
                            ):
                                self.groups[MeshAxis.EP] = group
            for ep_local_idx in range(self.mesh.ep):
                dp_indices = list(range(ep_local_idx, self.mesh.dp, self.mesh.ep))
                for p in range(self.mesh.pp):
                    for c in range(self.mesh.cp):
                        for t in range(self.mesh.tp):
                            group_ranks = [int(ranks[d, p, c, t]) for d in dp_indices]
                            group = dist.new_group(group_ranks)
                            if (
                                dp_idx % self.mesh.ep == ep_local_idx
                                and p == pp_idx
                                and c == cp_idx
                                and t == tp_idx
                                and len(group_ranks) > 1
                            ):
                                self.groups[MeshAxis.EDP] = group
        for d in range(self.mesh.dp):
            for p in range(self.mesh.pp):
                for t in range(self.mesh.tp):
                    group_ranks = ranks[d, p, :, t].tolist()
                    group = dist.new_group(group_ranks)
                    if d == dp_idx and p == pp_idx and t == tp_idx and self.mesh.cp > 1:
                        self.groups[MeshAxis.CP] = group
        for d in range(self.mesh.dp):
            for c in range(self.mesh.cp):
                for t in range(self.mesh.tp):
                    group_ranks = ranks[d, :, c, t].tolist()
                    group = dist.new_group(group_ranks)
                    if d == dp_idx and c == cp_idx and t == tp_idx and self.mesh.pp > 1:
                        self.groups[MeshAxis.PP] = group
        for p in range(self.mesh.pp):
            for c in range(self.mesh.cp):
                for t in range(self.mesh.tp):
                    group_ranks = ranks[:, p, c, t].tolist()
                    group = dist.new_group(group_ranks)
                    if p == pp_idx and c == cp_idx and t == tp_idx and self.mesh.dp > 1:
                        self.groups[MeshAxis.DP] = group

    def get_group(self, axis: MeshAxis) -> dist.ProcessGroup | None:
        return self.groups.get(axis)
