from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch.distributed as dist

from train_system.parallel.plan import MeshAxis, ParallelPlan, ProcessMesh


@dataclass
class MeshRuntime:
    mesh: ProcessMesh
    groups: dict[MeshAxis, dist.ProcessGroup | None] = field(default_factory=dict)

    @classmethod
    def from_plan(cls, plan: ParallelPlan) -> "MeshRuntime":
        runtime = cls(mesh=plan.mesh)
        runtime.initialize_groups()
        return runtime

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
