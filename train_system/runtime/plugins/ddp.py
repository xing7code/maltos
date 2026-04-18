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
            handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=self.ddp_group, async_op=True))
        for handle in handles:
            handle.wait()