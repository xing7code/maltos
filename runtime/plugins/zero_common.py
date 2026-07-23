from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch
import torch.distributed as dist
import torch.nn as nn

from runtime.types import ParamRole
from runtime.mesh import MeshAxis
from runtime.plugin import PluginId, RuntimePlugin


class ChainedWork:
    """A single grad-sync chain: 0..N async collectives fired back-to-back.

    Each link's functor fires its own collective (or does a purely local
    tensor op and returns a completed no-op Work) and returns a `dist.Work`.
    `fire()` walks the chain from the root. Between consecutive links it
    needs to guarantee the next collective observes the data the previous
    one wrote:

    - On stream-based (NCCL) backends this is a `block_current_stream()`
      call -- a non-blocking insert of a CUDA stream dependency.
    - Gloo has no stream/event concept (`block_current_stream()` raises
      `RuntimeError: Failed to create StreamBlock` on a CPU-only Work), so
      a gloo parent link must use a real, blocking `.wait()` instead. This
      matches gloo's own async_op semantics (a background thread already
      does the blocking internally), so it doesn't give up any overlap
      gloo didn't already lack.

    Pass `blocks_by_stream=False` for a functor whose collective runs on
    gloo so its link uses `.wait()` to synchronize with the next link
    instead of `block_current_stream()`.

    `wait()` only needs to wait on the last link: by the time it completes,
    everything upstream has completed too (guaranteed by the stream
    dependency chain on NCCL, or by the blocking wait on gloo).

    This replaces the old pattern of bespoke per-collective wrapper classes
    and the background-thread post-reduction worker: any per-collective
    post-processing (e.g. multiplying by a correction factor) belongs
    *inside* a functor as a pre-communication local op, not as a wait-time
    step, so a chain of arbitrary length still ends in a single `dist.Work`
    to wait on. See `CompletedWork`/`LocalCopyWork`/`NarrowShardWork` below
    for the small set of reusable wait-time tensor ops.
    """

    def __init__(
        self,
        parent: "ChainedWork | None",
        functor: Callable[[], dist.Work],
        *,
        blocks_by_stream: bool = True,
    ) -> None:
        self.parent = parent
        self.functor = functor
        self.blocks_by_stream = blocks_by_stream
        self.handle: dist.Work | None = None

    def fire(self) -> None:
        if self.parent is not None:
            # A ChainedWork may be persistent and re-fired every training step
            # (EP buckets do this). Each top-level fire must therefore replay
            # the entire chain; a non-None handle only describes the previous
            # invocation and must not suppress the parent's new collective.
            self.parent.fire()
            assert self.parent.handle is not None
            if self.parent.blocks_by_stream:
                self.parent.handle.block_current_stream()
            else:
                self.parent.handle.wait()
        self.handle = self.functor()

    def wait(self) -> None:
        if self.handle is None:
            self.fire()
        assert self.handle is not None
        self.handle.wait()


class CompletedWork:
    """dist.Work stand-in for a step that already finished synchronously (or did nothing)."""

    def wait(self) -> None:
        pass

    def block_current_stream(self) -> None:
        pass


class LocalCopyWork(CompletedWork):
    """Single-rank (no collective) case: just copy the local shard synchronously."""

    def __init__(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        dst.copy_(src)


class NarrowShardWork:
    """Wraps a full-buffer all-reduce Work; narrows out this rank's shard once
    the all-reduce completes and writes it into local_grad.

    Pass `accumulate=True` (ZeRO-2/3, where multiple exec_states/microbatches
    each contribute additively to the same local_param.grad) to add the shard
    in; leave it False (ZeRO-1, one reduction per bucket) to overwrite.
    """

    def __init__(
        self,
        work: dist.Work,
        grad_buffer: torch.Tensor,
        local_grad: torch.Tensor,
        shard_start: int,
        shard_end: int,
        *,
        accumulate: bool = False,
    ) -> None:
        self.work = work
        self.grad_buffer = grad_buffer
        self.local_grad = local_grad
        self.shard_start = shard_start
        self.shard_end = shard_end
        self.accumulate = accumulate

    def block_current_stream(self) -> None:
        self.work.block_current_stream()

    def wait(self) -> None:
        self.work.wait()
        shard = self.grad_buffer[self.shard_start : self.shard_end]
        if self.accumulate:
            self.local_grad.add_(shard)
        else:
            self.local_grad.copy_(shard)


def expert_erep_correction(*, tp: int, cp: int, ep: int, reuse_tp: bool, reuse_cp: bool) -> float:
    """Multiplier to turn an AVG all-reduce over MeshAxis.EREP into the correct op.

    EREP mixes two different kinds of replication depending on how EP reuses
    other axes: seq-slice multiplicity (when reuse_tp/reuse_cp let EP fall
    inside TP/CP without full sequence gathering, so different EREP members
    hold disjoint token subsets and their expert grads must be *summed*) and
    genuine DP-style data replication (which must be *averaged*). AVG divides
    by EREP world size unconditionally; multiplying by this correction after
    AVG turns the seq-slice portion of that division back into a sum while
    leaving any real DP averaging in place.
    """
    if reuse_tp and reuse_cp:
        return float(tp * cp // ep) if ep <= tp * cp else 1.0
    if reuse_tp and not reuse_cp:
        return float(tp // ep) if ep <= tp else 1.0
    if not reuse_tp and reuse_cp:
        return float(cp // ep) if ep <= cp else 1.0
    return 1.0


def rearm_bucket_pending(num_params: int, *, grad_accum_end: bool) -> int:
    """Pending-hook-count to arm a bucket with at the start of a backward.

    Grad accumulation reuses the same grad buffer across multiple backward
    micro-steps without zeroing it between them (only at the first micro-step
    of the window), so intermediate micro-steps must NOT trigger the bucket's
    collective -- only the final micro-step (grad_accum_end, i.e.
    StepContext.is_step_boundary) should. Arming pending to 0 on intermediate
    steps means each param's hook decrements it negative (-1, -2, ...),
    never hitting the `pending == 0` check that fires the collective; only
    when armed to num_params (on the step-boundary micro-step) does the
    hook's final decrement land exactly on 0.
    """
    return num_params if grad_accum_end else 0


def build_param_buckets(params: list[nn.Parameter], bucket_byte_size: int) -> list[list[nn.Parameter]]:
    """Greedily groups params (in the given order) into buckets no larger than
    bucket_byte_size, so a plugin can coalesce many small collectives into one
    per bucket instead of firing one collective per parameter."""
    buckets: list[list[nn.Parameter]] = []
    curr_bucket: list[nn.Parameter] = []
    curr_bytes = 0
    for param in params:
        param_bytes = param.numel() * param.element_size()
        if curr_bucket and curr_bytes + param_bytes > bucket_byte_size:
            buckets.append(curr_bucket)
            curr_bucket = []
            curr_bytes = 0
        curr_bucket.append(param)
        curr_bytes += param_bytes
    if curr_bucket:
        buckets.append(curr_bucket)
    return buckets


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


class ZeroPluginBase(RuntimePlugin):
    def __init__(
        self,
        *,
        id: PluginId,
        name: str,
        owns_optimizer: bool,
        owns_model_state: bool = False,
        runs_after: set[PluginId],
        bucket_mb_size: int = 25,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            owns_optimizer=owns_optimizer,
            owns_model_state=owns_model_state,
            runs_after=runs_after,
        )
        self.bucket_byte_size = bucket_mb_size * 1024 * 1024
        self.dp_group: dist.ProcessGroup | None = None
        self.world_size = 1
        self.rank = 0
        self.buckets: list = []
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._chained_work_wrappers: list[
            tuple[Callable[[ChainedWork, torch.Tensor], ChainedWork], ParamRole | None]
        ] = []

    def optimizer_state_source_rank(self, rank_id: int) -> int:
        return rank_id

    def annotate_param_metadata(self) -> None:
        return

    def wrap_chained_work(
        self,
        wrap: Callable[[ChainedWork, torch.Tensor], ChainedWork],
        *,
        role_filter: ParamRole | None = None,
    ) -> None:
        """Append a reduction step to future bucket work chains."""
        self._chained_work_wrappers.append((wrap, role_filter))

    def _apply_chained_work_wrappers(
        self,
        work: ChainedWork,
        tensor: torch.Tensor,
        role: ParamRole,
    ) -> ChainedWork:
        for wrap, role_filter in self._chained_work_wrappers:
            if role_filter is None or role_filter == role:
                work = wrap(work, tensor)
        return work

    def _group_context_for_role(self, role: ParamRole) -> GroupContext:
        assert self.runtime is not None
        mesh = self.runtime.mesh
        if role == ParamRole.EXPERT:
            group = self.runtime.get_group(MeshAxis.EREP)
            if group is None:
                return GroupContext(None, 1, 0)
            plan = self.runtime.plan
            correction = expert_erep_correction(
                tp=mesh.tp,
                cp=mesh.cp,
                ep=mesh.ep,
                reuse_tp=getattr(plan, 'reuse_tp_for_ep', True),
                reuse_cp=getattr(plan, 'reuse_cp_for_ep', True),
            )
            return GroupContext(group, dist.get_world_size(group), dist.get_rank(group), correction=correction)
        # SHARED: shard over DCP (DP × CP); AVG divides by dp*cp, want to divide by dp only.
        dcp_group = self.runtime.get_group(MeshAxis.DCP)
        if dcp_group is None:
            assert self.dp_group is not None
            return GroupContext(self.dp_group, self.world_size, self.rank)
        return GroupContext(dcp_group, dist.get_world_size(dcp_group), dist.get_rank(dcp_group), correction=float(mesh.cp))

    def _padded_len(self, numel: int, world_size: int) -> int:
        return (numel + world_size - 1) // world_size * world_size

    def _role_params(self, model: nn.Module, role: ParamRole) -> list[nn.Parameter]:
        assert self.runtime is not None
        return [
            param
            for param in model.parameters()
            if param.requires_grad and self.runtime.get_param_role(param) == role
        ][::-1]

    def _build_param_buckets(self, params: list[nn.Parameter]) -> list[list[nn.Parameter]]:
        return build_param_buckets(params, self.bucket_byte_size)

    def _maybe_clip_local_shards(self) -> None:
        assert self.runtime is not None
        max_norm = self.runtime.grad_clip_max_norm
        if max_norm is None or not self.buckets:
            return
        grad_device = None
        for bucket in self.buckets:
            grad = getattr(bucket.local_param, "grad", None)
            if grad is not None:
                grad_device = grad.device
                break
        if grad_device is None:
            return

        # A 13B model can overflow a float32 sum of otherwise finite squared
        # gradients. Keep the norm accumulator in float64; NCCL supports the
        # corresponding all-reduce and this runs once per optimizer step.
        local_sq = torch.zeros((), dtype=torch.float64, device=grad_device)
        for bucket in self.buckets:
            local_sq.add_(self._bucket_local_sq(bucket))
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
        global_norm = float(local_sq.sqrt().item())
        if not math.isfinite(global_norm):
            bad_buckets = []
            for bucket in self.buckets:
                grad = getattr(bucket.local_param, "grad", None)
                if grad is None or torch.isfinite(grad).all():
                    continue
                names = list(getattr(bucket, "logical_names", ()))
                bad_buckets.append(
                    f"bucket={getattr(bucket, 'index', '?')} names={names[:4]}"
                )
            details = "; ".join(bad_buckets[:8]) or "no local non-finite shard found"
            raise FloatingPointError(
                "non-finite ZeRO gradient norm; refusing optimizer step to avoid inf * 0 -> NaN. "
                f"Affected local shards: {details}"
            )
        clip_coef = float(max_norm) / (global_norm + 1e-6)
        if clip_coef < 1.0:
            for bucket in self.buckets:
                grad = getattr(bucket.local_param, "grad", None)
                if grad is not None:
                    grad.mul_(clip_coef)
        self.runtime.state.metadata["grad_norm"] = global_norm

    def _bucket_local_sq(self, bucket) -> torch.Tensor:
        raise NotImplementedError

    def collect_metrics(self) -> dict[str, float]:
        assert self.runtime is not None
        grad_norm = self.runtime.state.metadata.get("grad_norm")
        max_norm = self.runtime.grad_clip_max_norm
        if grad_norm is None or max_norm is None:
            return {}
        return {"grad_norm": float(grad_norm), "max_norm": float(max_norm)}
