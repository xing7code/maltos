from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime.core import RuntimeCore

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn

from parallel.expert_interfaces import ExpertParallelMoEModule
from runtime.mesh import MeshAxis
from runtime.plugin import ExpertParallelizableModule, PluginId, RuntimePlugin
from runtime.plugins.zero_common import ChainedWork, CompletedWork, build_param_buckets, expert_erep_correction, rearm_bucket_pending
from runtime.types import ParamRole, RuntimePhase


@dataclass
class _GradBucket:
    """Coalesces several expert params' grads into one shared buffer so their
    EREP all-reduce (+ any wrap applied via wrap_chained_work, e.g. CP sync)
    fires once per bucket instead of once per parameter.

    `work` is a single persistent ChainedWork built once (any wrap applied
    once, at bind time via wrap_chained_work) and re-fired every step:
    ChainedWork.fire() re-invokes its functors each call, and grad_buffer is
    a fixed tensor that's zeroed/refilled per step, so re-firing the same
    chain object is safe and avoids rebuilding the chain every microstep.
    `fired` tracks whether this step's hook has already fired it (replacing
    the old "work is None" sentinel, since work is never None now).
    """

    params: list[nn.Parameter]
    grad_buffer: torch.Tensor
    pending: int
    # Assigned immediately after construction (needs `self` to close over in
    # its functor) -- never None once _prepare_grad_buckets returns.
    work: ChainedWork = field(default=None)  # type: ignore[assignment]
    fired: bool = False


@dataclass(frozen=True)
class _MoEMetadata:
    dim: int
    num_experts: int


class _ExpertParallelMoE(nn.Module):
    def __init__(
        self,
        *,
        router: nn.Module,
        local_experts: nn.ModuleList,
        local_expert_ids: list[int],
        metadata: _MoEMetadata,
        ep_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self.router = router
        self.local_experts = local_experts
        self.local_expert_ids = tuple(local_expert_ids)
        self.hidden_size = metadata.dim
        self.num_experts = metadata.num_experts
        self.ep_group = ep_group
        self.ep_rank = dist.get_rank(ep_group)
        self.ep_world_size = dist.get_world_size(ep_group)
        if self.num_experts % self.ep_world_size != 0:
            raise ValueError(
                "ExpertParallelPlugin requires num_experts divisible by ep size, "
                f"got num_experts={self.num_experts}, ep={self.ep_world_size}"
            )
        if len(self.local_expert_ids) * self.ep_world_size != self.num_experts:
            raise ValueError("ExpertParallelPlugin expected evenly sharded local experts")

    @classmethod
    def from_moe(cls, moe: ExpertParallelMoEModule, ep_group: dist.ProcessGroup) -> "_ExpertParallelMoE":
        num_experts = int(moe.num_experts)
        experts = moe.experts
        if len(experts) != num_experts:
            raise ValueError(f"ExpertParallelPlugin expected len(experts)==num_experts, got {len(experts)} vs {num_experts}")
        ep_rank = dist.get_rank(ep_group)
        ep_world_size = dist.get_world_size(ep_group)
        experts_per_rank = num_experts // ep_world_size
        start = ep_rank * experts_per_rank
        end = start + experts_per_rank
        local_ids = list(range(start, end))
        local_experts = nn.ModuleList([experts[idx] for idx in local_ids])
        sample_expert = local_experts[0]
        hidden_size = int(moe.hidden_size)
        dim = int(moe.dim)
        if dim <= 0 or hidden_size <= 0:
            raise ValueError("ExpertParallelPlugin requires positive moe.dim and moe.hidden_size")
        return cls(
            router=moe.router,
            local_experts=local_experts,
            local_expert_ids=local_ids,
            metadata=_MoEMetadata(dim=dim, num_experts=num_experts),
            ep_group=ep_group,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = x.shape
        flat = x.reshape(-1, hidden_size)
        router_logits = self.router(flat)
        router_probs = router_logits.softmax(dim=-1)
        expert_idx = router_probs.argmax(dim=-1)
        expert_weight = router_probs.gather(1, expert_idx.unsqueeze(1)).squeeze(1)
        experts_per_rank = len(self.local_expert_ids)
        dest_rank = torch.div(expert_idx, experts_per_rank, rounding_mode="floor")
        local_expert_idx = expert_idx - dest_rank * experts_per_rank

        order = torch.argsort(dest_rank)
        send_tokens = flat.index_select(0, order).contiguous()
        send_weights = expert_weight.index_select(0, order).contiguous()
        send_local_expert_idx = local_expert_idx.index_select(0, order).to(dtype=torch.int64).contiguous()
        send_counts = torch.bincount(dest_rank, minlength=self.ep_world_size).to(dtype=torch.int64)
        recv_counts = _exchange_counts(send_counts, self.ep_group)
        send_split_sizes = send_counts.tolist()
        recv_split_sizes = recv_counts.tolist()

        recv_total = int(recv_counts.sum().item())
        recv_tokens = torch.empty((recv_total, hidden_size), dtype=flat.dtype, device=flat.device)
        recv_weights = torch.empty((recv_total,), dtype=expert_weight.dtype, device=flat.device)
        recv_local_expert_idx = torch.empty((recv_total,), dtype=torch.int64, device=flat.device)

        recv_tokens = dist_nn.all_to_all_single(
            recv_tokens,
            send_tokens,
            output_split_sizes=recv_split_sizes,
            input_split_sizes=send_split_sizes,
            group=self.ep_group,
        )
        recv_weights = dist_nn.all_to_all_single(
            recv_weights,
            send_weights,
            output_split_sizes=recv_split_sizes,
            input_split_sizes=send_split_sizes,
            group=self.ep_group,
        )
        recv_local_expert_idx = dist_nn.all_to_all_single(
            recv_local_expert_idx,
            send_local_expert_idx,
            output_split_sizes=recv_split_sizes,
            input_split_sizes=send_split_sizes,
            group=self.ep_group,
        )

        recv_outputs = torch.zeros_like(recv_tokens)
        for local_idx, expert in enumerate(self.local_experts):
            mask = recv_local_expert_idx == local_idx
            if not torch.any(mask):
                continue
            expert_out = expert(recv_tokens[mask]) * recv_weights[mask].unsqueeze(1)
            recv_outputs[mask] = expert_out.to(recv_outputs.dtype)

        returned_outputs = torch.empty_like(send_tokens)
        returned_outputs = dist_nn.all_to_all_single(
            returned_outputs,
            recv_outputs,
            output_split_sizes=send_split_sizes,
            input_split_sizes=recv_split_sizes,
            group=self.ep_group,
        )

        out = torch.zeros_like(flat)
        out.index_copy_(0, order, returned_outputs)
        return out.view(batch, seq_len, hidden_size)


def _exchange_counts(send_counts: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    recv_counts = torch.empty_like(send_counts)
    split_sizes = [1] * send_counts.numel()
    dist.all_to_all_single(
        recv_counts,
        send_counts.contiguous(),
        output_split_sizes=split_sizes,
        input_split_sizes=split_sizes,
        group=group,
    )
    return recv_counts


class ExpertParallelPlugin(RuntimePlugin):
    def __init__(self, bucket_mb_size: int = 25) -> None:
        super().__init__(
            id=PluginId.EP,
            name="expert_parallel",
            runs_after={PluginId.TP, PluginId.SP},
            runs_before={PluginId.DP, PluginId.ZERO1, PluginId.ZERO2, PluginId.ZERO3},
        )
        self.bucket_byte_size = bucket_mb_size * 1024 * 1024
        self._expert_param_ids: set[int] = set()
        self._shared_grad_sync_handles: list[dist.Work] = []
        self._delegate_shared_dp_sync = False
        self._delegate_expert_sync = False
        # Only populated when EP handles expert-grad sync itself (no ZeRO
        # active): coalesces expert param grads into a handful of buckets so
        # POST_BACKWARD fires one EREP all-reduce per bucket, not per param.
        self._grad_buckets: list[_GradBucket] = []

    def wrap_chained_work(self, wrap: Callable[[ChainedWork, torch.Tensor], ChainedWork]) -> None:
        """Layer one more sync step onto every expert-grad bucket's chain.

        Must be called after EP's transform_model has built its buckets
        (e.g. from another plugin's annotate_param_metadata, not bind) --
        raises if called before any buckets exist. `wrap(work, grad_buffer)`
        receives a bucket's current `ChainedWork` (its EREP all-reduce, or
        any previously applied wrap) and the bucket's grad buffer tensor,
        and must return a new `ChainedWork` -- typically `ChainedWork(work,
        lambda: my_collective(grad_buffer), blocks_by_stream=...)`. Applied
        immediately, directly, to every existing bucket's `.work`. Callable
        any number of times -- each call layers one more step on top of
        whatever is already there, in call order; there's no hidden
        registration list to reorder, so composing multiple wraps is safe.
        """
        if not self._grad_buckets:
            raise RuntimeError(
                "ExpertParallelPlugin.wrap_chained_work called before grad buckets exist "
                "-- call from annotate_param_metadata (after transform_model), not bind"
            )
        for bucket in self._grad_buckets:
            bucket.work = wrap(bucket.work, bucket.grad_buffer)

    @property
    def ep_group(self) -> dist.ProcessGroup:
        assert self.runtime is not None
        group = self.runtime.get_group(MeshAxis.EP)
        if group is None:
            raise ValueError("ExpertParallelPlugin requires an EP process group")
        return group

    @property
    def dp_group(self) -> dist.ProcessGroup | None:
        assert self.runtime is not None
        return self.runtime.get_group(MeshAxis.DP)

    @property
    def edp_group(self) -> dist.ProcessGroup | None:
        assert self.runtime is not None
        return self.runtime.get_group(MeshAxis.EREP)

    def bind(self, runtime: "RuntimeCore") -> None:
        super().bind(runtime)
        active = {plugin.id for plugin in runtime.plugins if plugin is not self}
        self._delegate_shared_dp_sync = bool({PluginId.DP, PluginId.ZERO1, PluginId.ZERO2, PluginId.ZERO3} & active)
        self._delegate_expert_sync = bool({PluginId.ZERO1, PluginId.ZERO2, PluginId.ZERO3} & active)
        self._validate_runtime_support()

    def transform_model(self, model: nn.Module) -> nn.Module:
        if not isinstance(model, ExpertParallelizableModule):
            raise TypeError(
                "ExpertParallelPlugin requires model.expert_parallel_spec(), "
                f"got {type(model).__name__}"
            )
        spec = model.expert_parallel_spec()
        for path in spec.moe_paths:
            if self.runtime.is_module_path_omitted(path):
                continue
            try:
                module = model.get_submodule(path)
            except AttributeError:
                raise
            _validate_supported_moe_module(module)
            model.set_submodule(path, _ExpertParallelMoE.from_moe(module, self.ep_group))
        self._expert_param_ids = {
            id(param)
            for module in model.modules()
            if isinstance(module, _ExpertParallelMoE)
            for param in module.local_experts.parameters()
        }
        for param in model.parameters():
            if not param.requires_grad:
                continue
            role = ParamRole.EXPERT if id(param) in self._expert_param_ids else ParamRole.SHARED
            self.runtime.set_param_role(param, role)
        if not self._delegate_expert_sync:
            self._prepare_grad_buckets(model)
        return model

    def _prepare_grad_buckets(self, model: nn.Module) -> None:
        expert_params = [
            param
            for param in model.parameters()
            if param.requires_grad and id(param) in self._expert_param_ids
        ]
        if not expert_params:
            return
        dtype, device = expert_params[0].dtype, expert_params[0].device
        edp_group = self.edp_group
        edp_blocks_by_stream = edp_group is None or dist.get_backend(edp_group) != "gloo"
        for bucket_params in build_param_buckets(expert_params, self.bucket_byte_size):
            grad_buffer = torch.zeros(sum(p.numel() for p in bucket_params), dtype=dtype, device=device)
            offset = 0
            for param in bucket_params:
                param.grad = grad_buffer[offset : offset + param.numel()].view_as(param)
                offset += param.numel()
            bucket = _GradBucket(params=bucket_params, grad_buffer=grad_buffer, pending=0)
            bucket.work = ChainedWork(None, self._bucket_erep_functor(bucket, edp_group), blocks_by_stream=edp_blocks_by_stream)
            self._grad_buckets.append(bucket)
            for param in bucket_params:
                param.register_post_accumulate_grad_hook(self._make_bucket_grad_hook(bucket))

    def _make_bucket_grad_hook(self, bucket: _GradBucket):
        def hook(_param: nn.Parameter) -> None:
            bucket.pending -= 1
            if bucket.pending == 0:
                bucket.work.fire()
                bucket.fired = True

        return hook

    def is_expert_param(self, param: nn.Parameter) -> bool:
        assert self.runtime is not None
        return self.runtime.get_param_role(param) == ParamRole.EXPERT

    def annotate_param_metadata(self) -> None:
        assert self.runtime is not None
        erep_group = self.edp_group
        erep_size = dist.get_world_size(erep_group) if erep_group is not None else 1
        if erep_size <= 1:
            return
        for fq_name, _ in self.runtime.state_manager.iter_param_states():
            param = self.runtime.state_manager.get_param_tensor(fq_name)
            if self.runtime.get_param_role(param) != ParamRole.EXPERT:
                continue
            self.runtime.state_manager.add_param_replicated_axis(fq_name, MeshAxis.EREP)

    def on_phase(self, phase: RuntimePhase) -> None:
        assert self.runtime is not None
        if phase == RuntimePhase.PRE_BACKWARD:
            if not self._delegate_expert_sync and self.runtime.state.step_context.accum_start:
                for bucket in self._grad_buckets:
                    bucket.grad_buffer.zero_()
                    offset = 0
                    for param in bucket.params:
                        param.grad = bucket.grad_buffer[
                            offset : offset + param.numel()
                        ].view_as(param)
                        offset += param.numel()
            if self.runtime.state.step_context.backward_start:
                grad_accum_end = self.runtime.state.step_context.is_step_boundary
                for bucket in self._grad_buckets:
                    bucket.pending = rearm_bucket_pending(len(bucket.params), grad_accum_end=grad_accum_end)
                    bucket.fired = False
            return
        if phase != RuntimePhase.POST_BACKWARD:
            return
        if not self.runtime.state.step_context.is_step_boundary:
            return
        # See Zero1Plugin.on_phase: run_step() callers may read .grad once it
        # returns, so all grad-sync work below must be fired AND waited here.
        if not self._delegate_expert_sync:
            self._wait_expert_grad_sync()
        if self._delegate_shared_dp_sync:
            return
        if self.dp_group is None or dist.get_world_size(self.dp_group) <= 1:
            return
        self._shared_grad_sync_handles.clear()
        for param in self.runtime.model.parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if id(param) in self._expert_param_ids:
                continue
            self._shared_grad_sync_handles.append(
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=self.dp_group, async_op=True)
            )
        for handle in self._shared_grad_sync_handles:
            handle.wait()
        self._shared_grad_sync_handles.clear()

    def _wait_expert_grad_sync(self) -> None:
        # Every rank must fire the same sequence of collectives on a bucket's
        # group, even if this rank's hook never ran for it (e.g. a PP stage
        # that doesn't own these experts this microbatch) -- otherwise a peer
        # rank that DID fire waits forever on this one.
        for bucket in self._grad_buckets:
            if not bucket.fired:
                bucket.work.fire()
                bucket.fired = True
            bucket.work.wait()

    def _bucket_erep_functor(self, bucket: _GradBucket, edp_group: dist.ProcessGroup | None):
        assert self.runtime is not None

        def functor() -> dist.Work:
            if edp_group is None or dist.get_world_size(edp_group) <= 1:
                return CompletedWork()
            mesh = self.runtime.mesh
            plan = self.runtime.plan
            correction = expert_erep_correction(
                tp=mesh.tp,
                cp=mesh.cp,
                ep=mesh.ep,
                reuse_tp=getattr(plan, "reuse_tp_for_ep", True),
                reuse_cp=getattr(plan, "reuse_cp_for_ep", True),
            )
            if correction != 1.0:
                bucket.grad_buffer.mul_(correction)
            return dist.all_reduce(bucket.grad_buffer, op=dist.ReduceOp.AVG, group=edp_group, async_op=True)

        return functor

    def _validate_runtime_support(self) -> None:
        assert self.runtime is not None
        mesh = self.runtime.mesh
        if mesh.ep <= 1:
            raise ValueError("ExpertParallelPlugin requires mesh.ep > 1")
        if mesh.pp < 1 or mesh.cp < 1:
            raise ValueError(
                "ExpertParallelPlugin requires pp>=1 and cp>=1, "
                f"got dp={mesh.dp} tp={mesh.tp} pp={mesh.pp} cp={mesh.cp} ep={mesh.ep}"
            )
        active = {plugin.id for plugin in self.runtime.plugins if plugin is not self}


def _validate_supported_moe_module(module: nn.Module) -> None:
    if not isinstance(module, ExpertParallelMoEModule):
        raise TypeError(
            "ExpertParallelPlugin requires MoE modules to satisfy ExpertParallelMoEModule, "
            f"got module type={type(module).__name__}"
        )
