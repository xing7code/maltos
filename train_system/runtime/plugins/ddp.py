from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from train_system.runtime.plugin import BaseParallelPlugin


class NaiveDdpPlugin(BaseParallelPlugin):

    def __init__(self, ddp_group: dist.ProcessGroup):
        self.ddp_group = ddp_group
        self.world_size = dist.get_world_size(self.ddp_group)
        self.rank = dist.get_rank(self.ddp_group)

    def after_backward(self, model: nn.Module):
        for k, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert p.grad is not None, f"param {k} has requires_grad=True but grad is None"
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=self.ddp_group)


class NaiveAsyncDdpPlugin(BaseParallelPlugin):

    def __init__(self, ddp_group: dist.ProcessGroup):
        self.ddp_group = ddp_group
        self.world_size = dist.get_world_size(self.ddp_group)
        self.rank = dist.get_rank(self.ddp_group)

    def after_backward(self, model: nn.Module):
        handles = []
        for k, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert p.grad is not None, f"param {k} has requires_grad=True but grad is None"
            handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=self.ddp_group, async_op=True))
        for handle in handles:
            handle.wait()


@dataclass
class _Bucket:
    group: dist.ProcessGroup
    params: list[nn.Parameter] = field(default_factory=list)
    param_views: list[torch.Tensor] = field(default_factory=list)
    flat_buffer: torch.Tensor | None = None
    total_bytes: int = 0
    pending: int = 0
    handle: dist.Work | None = None

    def add_param(self, p):
        self.params.append(p)
        self.total_bytes += p.numel() * p.element_size()

    def finalize(self):
        total_numel = sum(p.numel() for p in self.params)
        dtype, device = self.params[0].dtype, self.params[0].device
        self.flat_buffer = torch.zeros(total_numel, dtype=dtype, device=device)
        offset = 0
        for p in self.params:
            p_view = self.flat_buffer[offset : offset+p.numel()].view_as(p)
            self.param_views.append(p_view)
            offset += p.numel()
        self.reset()

    def reset(self):
        self.pending = len(self.params)
        self.handle = None
        # This is needed every round of backward, incase zero_grad called to set p.grad to None.
        self.flat_buffer.zero_()
        for p, view in zip(self.params, self.param_views):
            p.grad = view

    def _sync_grad(self):
        self.handle = dist.all_reduce(self.flat_buffer, op=dist.ReduceOp.AVG, group=self.group, async_op=True)

    def make_hook(self):
        def hook(p):
            self.pending -= 1
            if self.pending==0:
                self._sync_grad()
        return hook
    

class DdpWithBucketPlugin(BaseParallelPlugin):

    def __init__(self, ddp_group: dist.ProcessGroup, bucket_mb_size: int):
        self.ddp_group = ddp_group
        self.world_size = dist.get_world_size(ddp_group)
        self.rank = dist.get_rank(ddp_group)
        self.bucket_byte_size = bucket_mb_size * 1024 * 1024
        self.buckets = None

    def _build_buckets(self, model):
        self.buckets = []

        for p in reversed(list(model.parameters())):
            if not p.requires_grad:
                continue
            if not self.buckets or self.buckets[-1].total_bytes >= self.bucket_byte_size:
                if self.buckets:
                    self.buckets[-1].finalize()
                self.buckets.append(_Bucket(self.ddp_group))
            self.buckets[-1].add_param(p)
        self.buckets[-1].finalize()

    def setup_model(self, model: nn.Module) -> nn.Module:
        self._build_buckets(model)
        for bkt in self.buckets:
            for p in bkt.params:
                p.register_post_accumulate_grad_hook(bkt.make_hook())
        return model

    def before_forward(self, model: nn.Module):
        for bkt in self.buckets:
            bkt.reset()

    def after_backward(self, model: nn.Module):
        for bkt in self.buckets:
            assert bkt.handle is not None, f"bucket with {len(bkt.params)} params was never synced, likely some params have no grad"
            bkt.handle.wait()

    