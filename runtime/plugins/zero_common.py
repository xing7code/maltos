from __future__ import annotations

from dataclasses import dataclass
import threading

import torch
import torch.distributed as dist
import torch.nn as nn

from runtime.core import ParamRole
from runtime.mesh import MeshAxis
from runtime.plugin import PluginId, RuntimePlugin


@dataclass(frozen=True)
class GroupContext:
    group: dist.ProcessGroup | None
    world_size: int
    rank: int
    # Multiplier applied to local_grad after ReduceOp.AVG to correct for non-uniform
    # averaging semantics. AVG divides by world_size, but the desired divisor may differ
    # (e.g. only avg over DP, not CP). correction = world_size / desired_avg_factor.
    correction: float = 1.0

    @property
    def is_gloo(self) -> bool:
        return self.group is not None and dist.get_backend(self.group) == "gloo"

    @property
    def supports_async_overlap(self) -> bool:
        return self.group is not None and not self.is_gloo


class AllReduceShardWork:
    def __init__(
        self,
        work: dist.Work,
        grad_buffer: torch.Tensor,
        local_grad: torch.Tensor,
        shard_start: int,
        shard_end: int,
        correction: float = 1.0,
    ):
        self.work = work
        self.grad_buffer = grad_buffer
        self.local_grad = local_grad
        self.shard_start = shard_start
        self.shard_end = shard_end
        self.correction = correction

    def wait(self) -> None:
        self.work.wait()
        shard = self.grad_buffer[self.shard_start : self.shard_end]
        if self.correction != 1.0:
            shard = shard * self.correction
        self.local_grad.add_(shard)


class ReduceScatterShardWork:
    def __init__(
        self,
        work: dist.Work,
        shard_buffer: torch.Tensor,
        local_grad: torch.Tensor,
        correction: float = 1.0,
    ):
        self.work = work
        self.shard_buffer = shard_buffer
        self.local_grad = local_grad
        self.correction = correction

    def wait(self) -> None:
        self.work.wait()
        shard = self.shard_buffer if self.correction == 1.0 else self.shard_buffer * self.correction
        self.local_grad.add_(shard)


class _ZeroPluginBase(RuntimePlugin):
    def __init__(
        self,
        *,
        id: PluginId,
        name: str,
        owns_optimizer: bool,
        runs_after: set[PluginId],
        bucket_mb_size: int = 25,
    ) -> None:
        super().__init__(id=id, name=name, owns_optimizer=owns_optimizer, runs_after=runs_after)
        self.bucket_byte_size = bucket_mb_size * 1024 * 1024
        self.dp_group: dist.ProcessGroup | None = None
        self.world_size = 1
        self.rank = 0
        self.buckets: list = []
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._post_reduction_thread: threading.Thread | None = None
        self._post_reduction_cond = threading.Condition()

    def _group_context_for_role(self, role: ParamRole) -> GroupContext:
        assert self.runtime is not None
        mesh = self.runtime.mesh
        if role == ParamRole.EXPERT:
            group = self.runtime.get_group(MeshAxis.EREP)
            if group is None:
                return GroupContext(None, 1, 0)
            plan = self.runtime.plan
            reuse_tp = getattr(plan, 'reuse_tp_for_ep', True)
            reuse_cp = getattr(plan, 'reuse_cp_for_ep', True)
            if reuse_tp and reuse_cp:
                # EREP spans (TP*CP/EP) seq slots × DP data slots.
                # correction = TP*CP/EP when EP ≤ TP*CP (seq multiplicity); else 1.0.
                correction = float(mesh.tp * mesh.cp // mesh.ep) if mesh.ep <= mesh.tp * mesh.cp else 1.0
            elif reuse_tp and not reuse_cp:
                # EP groups per-CP; EREP spans TP/EP TP positions × DP data slots.
                # CP is not in EREP (groups are per-CP), so no CP factor.
                correction = float(mesh.tp // mesh.ep) if mesh.ep <= mesh.tp else 1.0
            elif not reuse_tp and reuse_cp:
                # EP groups per-TP; EREP spans CP/EP seq slots × DP data slots.
                correction = float(mesh.cp // mesh.ep) if mesh.ep <= mesh.cp else 1.0
            else:
                # EP uses DP only; EREP = pure DP replicas, no seq multiplicity.
                correction = 1.0
            return GroupContext(group, dist.get_world_size(group), dist.get_rank(group), correction=correction)
        # SHARED: shard over DCP (DP × CP); AVG divides by dp*cp, want to divide by dp only.
        dcp_group = self.runtime.get_group(MeshAxis.DCP)
        if dcp_group is None:
            assert self.dp_group is not None
            return GroupContext(self.dp_group, self.world_size, self.rank)
        return GroupContext(dcp_group, dist.get_world_size(dcp_group), dist.get_rank(dcp_group), correction=float(mesh.cp))

    def _padded_len(self, numel: int, world_size: int) -> int:
        return (numel + world_size - 1) // world_size * world_size

    def _use_async_worker(self) -> bool:
        return self.dp_group is not None and dist.get_backend(self.dp_group) != "gloo"

    def _start_post_reduction_worker(self) -> None:
        assert self.runtime is not None
        if not self.runtime._post_grad_reduction_callbacks:
            self._post_reduction_thread = None
            return
        self._post_reduction_thread = threading.Thread(
            target=self._post_reduction_worker, daemon=True
        )
        self._post_reduction_thread.start()

    def _role_params(self, model: nn.Module, role: ParamRole) -> list[nn.Parameter]:
        assert self.runtime is not None
        return [
            param
            for param in model.parameters()
            if param.requires_grad and self.runtime.get_param_role(param) == role
        ][::-1]

    def _build_param_buckets(self, params: list[nn.Parameter]) -> list[list[nn.Parameter]]:
        buckets: list[list[nn.Parameter]] = []
        curr_bucket: list[nn.Parameter] = []
        curr_bytes = 0
        for param in params:
            param_bytes = param.numel() * param.element_size()
            if curr_bucket and curr_bytes + param_bytes > self.bucket_byte_size:
                buckets.append(curr_bucket)
                curr_bucket = []
                curr_bytes = 0
            curr_bucket.append(param)
            curr_bytes += param_bytes
        if curr_bucket:
            buckets.append(curr_bucket)
        return buckets
