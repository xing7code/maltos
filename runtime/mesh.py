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
    DCP = "dcp"   # DP × CP replication group (non-expert ZeRO)
    EREP = "erep" # expert replication group: DP×CP / EP


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
        }.items():
            if size < 1:
                raise ValueError(f"{axis.value} must be >= 1, got {size}")


def _validate_ep_for_plan(mesh: MeshConfig, *, reuse_tp: bool, reuse_cp: bool) -> None:
    """Validate EP divisibility constraints given which dimensions EP is allowed to span."""
    ep, tp, cp, dp = mesh.ep, mesh.tp, mesh.cp, mesh.dp
    if ep == 1:
        return
    if reuse_tp and reuse_cp:
        # EP spans TP×CP×DP.
        space = tp * cp * dp
        if ep > space:
            raise ValueError(f"EP must be <= TP*CP*DP={space}, got ep={ep}")
        if space % ep != 0:
            raise ValueError(f"TP*CP*DP={space} must be divisible by EP={ep}")
        if ep <= tp:
            if tp % ep != 0:
                raise ValueError(f"When EP <= TP, TP must be divisible by EP, got tp={tp}, ep={ep}")
        elif ep <= tp * cp:
            if ep % tp != 0:
                raise ValueError(f"When TP < EP <= TP*CP, EP must be divisible by TP, got ep={ep}, tp={tp}")
            if cp % (ep // tp) != 0:
                raise ValueError(f"When TP < EP <= TP*CP, CP must be divisible by EP//TP, got cp={cp}, ep//tp={ep // tp}")
        else:
            if ep % (tp * cp) != 0:
                raise ValueError(f"When EP > TP*CP, EP must be divisible by TP*CP, got ep={ep}, tp*cp={tp * cp}")
            if dp % (ep // (tp * cp)) != 0:
                raise ValueError(f"When EP > TP*CP, DP must be divisible by EP//(TP*CP), got dp={dp}, ep//(tp*cp)={ep // (tp * cp)}")
    elif reuse_tp and not reuse_cp:
        # EP spans TP×DP per CP.
        space = tp * dp
        if ep > space:
            raise ValueError(f"With reuse_cp_for_ep=False, EP must be <= TP*DP={space}, got ep={ep}")
        if space % ep != 0:
            raise ValueError(f"With reuse_cp_for_ep=False, TP*DP={space} must be divisible by EP={ep}")
        if ep <= tp:
            if tp % ep != 0:
                raise ValueError(f"When EP <= TP, TP must be divisible by EP, got tp={tp}, ep={ep}")
        else:
            if ep % tp != 0:
                raise ValueError(f"When EP > TP, EP must be divisible by TP, got ep={ep}, tp={tp}")
            if dp % (ep // tp) != 0:
                raise ValueError(f"When EP > TP, DP must be divisible by EP//TP, got dp={dp}, ep//tp={ep // tp}")
    elif not reuse_tp and reuse_cp:
        # EP spans CP×DP per TP.
        space = cp * dp
        if ep > space:
            raise ValueError(f"With reuse_tp_for_ep=False, EP must be <= CP*DP={space}, got ep={ep}")
        if space % ep != 0:
            raise ValueError(f"With reuse_tp_for_ep=False, CP*DP={space} must be divisible by EP={ep}")
        if ep <= cp:
            if cp % ep != 0:
                raise ValueError(f"When EP <= CP, CP must be divisible by EP, got cp={cp}, ep={ep}")
        else:
            if ep % cp != 0:
                raise ValueError(f"When EP > CP, EP must be divisible by CP, got ep={ep}, cp={cp}")
            if dp % (ep // cp) != 0:
                raise ValueError(f"When EP > CP, DP must be divisible by EP//CP, got dp={dp}, ep//cp={ep // cp}")
    else:
        # EP spans DP only, per (TP, CP).
        if ep > dp:
            raise ValueError(f"With reuse_tp/cp_for_ep=False, EP must be <= DP={dp}, got ep={ep}")
        if dp % ep != 0:
            raise ValueError(f"With reuse_tp/cp_for_ep=False, DP={dp} must be divisible by EP={ep}")


@dataclass
class ProcessGroupManager:
    mesh: MeshConfig
    plan: "ParallelPlan | None" = None
    groups: dict[MeshAxis, dist.ProcessGroup | None] = field(default_factory=dict)

    @classmethod
    def from_mesh(cls, mesh: MeshConfig) -> "ProcessGroupManager":
        _validate_ep_for_plan(mesh, reuse_tp=True, reuse_cp=True)
        manager = cls(mesh=mesh)
        manager.initialize_groups()
        return manager

    @classmethod
    def from_plan(cls, plan: ParallelPlan, mesh: MeshConfig | None = None) -> "ProcessGroupManager":
        if mesh is None:
            raise ValueError("from_plan requires mesh because ParallelPlan does not own MeshConfig")
        _validate_ep_for_plan(mesh, reuse_tp=plan.reuse_tp_for_ep, reuse_cp=plan.reuse_cp_for_ep)
        manager = cls(mesh=mesh, plan=plan)
        manager.initialize_groups()
        return manager

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
            MeshAxis.DCP: None,
            MeshAxis.EREP: None,
        }
        if not dist.is_initialized():
            return

        global_rank = dist.get_rank()
        ranks = np.arange(self.world_size).reshape(self.mesh.dp, self.mesh.pp, self.mesh.cp, self.mesh.tp)
        dp_indices, pp_indices, cp_indices, tp_indices = np.where(ranks == global_rank)
        if len(dp_indices) != 1:
            raise ValueError(f"unable to locate global_rank={global_rank} in mesh ranks={ranks.tolist()}")
        dp_idx, pp_idx, cp_idx, tp_idx = int(dp_indices[0]), int(pp_indices[0]), int(cp_indices[0]), int(tp_indices[0])

        for d in range(self.mesh.dp):
            for p in range(self.mesh.pp):
                for c in range(self.mesh.cp):
                    group_ranks = ranks[d, p, c, :].tolist()
                    group = dist.new_group(group_ranks)
                    if d == dp_idx and p == pp_idx and c == cp_idx and self.mesh.tp > 1:
                        self.groups[MeshAxis.TP] = group

        if self.mesh.ep > 1:
            reuse_tp = self.plan.reuse_tp_for_ep if self.plan is not None else True
            reuse_cp = self.plan.reuse_cp_for_ep if self.plan is not None else True
            self._initialize_ep_groups(ranks, dp_idx, pp_idx, cp_idx, tp_idx, reuse_tp, reuse_cp)

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

        # DCP: all DP × CP ranks sharing the same (PP, TP) position.
        for p in range(self.mesh.pp):
            for t in range(self.mesh.tp):
                group_ranks = ranks[:, p, :, t].flatten().tolist()
                group = dist.new_group(group_ranks)
                if p == pp_idx and t == tp_idx and self.mesh.dp * self.mesh.cp > 1:
                    self.groups[MeshAxis.DCP] = group

    def _initialize_ep_groups(
        self,
        ranks: "np.ndarray",
        dp_idx: int,
        pp_idx: int,
        cp_idx: int,
        tp_idx: int,
        reuse_tp: bool,
        reuse_cp: bool,
    ) -> None:
        ep = self.mesh.ep
        tp = self.mesh.tp
        cp = self.mesh.cp
        dp = self.mesh.dp
        if reuse_tp and reuse_cp:
            # flat = t + c*TP + d*TP*CP  (TP innermost, global EP groups)
            my_flat = tp_idx + cp_idx * tp + dp_idx * tp * cp
            num_ep_groups = tp * cp * dp // ep
            for ep_group_id in range(num_ep_groups):
                for p in range(self.mesh.pp):
                    group_ranks = [
                        int(ranks[d, p, c, t])
                        for d in range(dp) for c in range(cp) for t in range(tp)
                        if (t + c * tp + d * tp * cp) // ep == ep_group_id
                    ]
                    group = dist.new_group(group_ranks)
                    if my_flat // ep == ep_group_id and p == pp_idx:
                        self.groups[MeshAxis.EP] = group
            for ep_local in range(ep):
                for p in range(self.mesh.pp):
                    group_ranks = [
                        int(ranks[d, p, c, t])
                        for d in range(dp) for c in range(cp) for t in range(tp)
                        if (t + c * tp + d * tp * cp) % ep == ep_local
                    ]
                    group = dist.new_group(group_ranks)
                    if my_flat % ep == ep_local and p == pp_idx and len(group_ranks) > 1:
                        self.groups[MeshAxis.EREP] = group
        elif reuse_tp and not reuse_cp:
            # flat per-CP: t + d*TP  (TP innermost, EP groups don't cross CP boundaries)
            my_flat_per_cp = tp_idx + dp_idx * tp
            num_ep_groups_per_cp = tp * dp // ep
            for c in range(cp):
                for ep_group_id in range(num_ep_groups_per_cp):
                    for p in range(self.mesh.pp):
                        group_ranks = [
                            int(ranks[d, p, c, t])
                            for d in range(dp) for t in range(tp)
                            if (t + d * tp) // ep == ep_group_id
                        ]
                        group = dist.new_group(group_ranks)
                        if c == cp_idx and my_flat_per_cp // ep == ep_group_id and p == pp_idx:
                            self.groups[MeshAxis.EP] = group
            for c in range(cp):
                for ep_local in range(ep):
                    for p in range(self.mesh.pp):
                        group_ranks = [
                            int(ranks[d, p, c, t])
                            for d in range(dp) for t in range(tp)
                            if (t + d * tp) % ep == ep_local
                        ]
                        group = dist.new_group(group_ranks)
                        if c == cp_idx and my_flat_per_cp % ep == ep_local and p == pp_idx and len(group_ranks) > 1:
                            self.groups[MeshAxis.EREP] = group
        elif not reuse_tp and reuse_cp:
            # flat per-TP: c + d*CP  (CP innermost, EP groups don't cross TP boundaries)
            my_flat_per_tp = cp_idx + dp_idx * cp
            num_ep_groups_per_tp = cp * dp // ep
            for t in range(tp):
                for ep_group_id in range(num_ep_groups_per_tp):
                    for p in range(self.mesh.pp):
                        group_ranks = [
                            int(ranks[d, p, c, t])
                            for d in range(dp) for c in range(cp)
                            if (c + d * cp) // ep == ep_group_id
                        ]
                        group = dist.new_group(group_ranks)
                        if t == tp_idx and my_flat_per_tp // ep == ep_group_id and p == pp_idx:
                            self.groups[MeshAxis.EP] = group
            for t in range(tp):
                for ep_local in range(ep):
                    for p in range(self.mesh.pp):
                        group_ranks = [
                            int(ranks[d, p, c, t])
                            for d in range(dp) for c in range(cp)
                            if (c + d * cp) % ep == ep_local
                        ]
                        group = dist.new_group(group_ranks)
                        if t == tp_idx and my_flat_per_tp % ep == ep_local and p == pp_idx and len(group_ranks) > 1:
                            self.groups[MeshAxis.EREP] = group
        else:
            # flat per-(TP,CP): d  (DP only, EP groups don't cross TP or CP)
            num_ep_groups_per_tcp = dp // ep
            for t in range(tp):
                for c in range(cp):
                    for ep_group_id in range(num_ep_groups_per_tcp):
                        for p in range(self.mesh.pp):
                            group_ranks = [
                                int(ranks[d, p, c, t])
                                for d in range(dp)
                                if d // ep == ep_group_id
                            ]
                            group = dist.new_group(group_ranks)
                            if t == tp_idx and c == cp_idx and dp_idx // ep == ep_group_id and p == pp_idx:
                                self.groups[MeshAxis.EP] = group
            for t in range(tp):
                for c in range(cp):
                    for ep_local in range(ep):
                        for p in range(self.mesh.pp):
                            group_ranks = [
                                int(ranks[d, p, c, t])
                                for d in range(dp)
                                if d % ep == ep_local
                            ]
                            group = dist.new_group(group_ranks)
                            if t == tp_idx and c == cp_idx and dp_idx % ep == ep_local and p == pp_idx and len(group_ranks) > 1:
                                self.groups[MeshAxis.EREP] = group

    def get_group(self, axis: MeshAxis) -> dist.ProcessGroup | None:
        return self.groups.get(axis)

    def close(self) -> None:
        if not dist.is_initialized():
            self.groups = {axis: None for axis in MeshAxis}
            return
        seen: set[int] = set()
        for group in self.groups.values():
            if group is None:
                continue
            group_id = id(group)
            if group_id in seen:
                continue
            seen.add(group_id)
            dist.destroy_process_group(group)
        self.groups = {axis: None for axis in MeshAxis}
