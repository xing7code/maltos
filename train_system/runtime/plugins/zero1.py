import torch
import torch.nn as nn
import torch.distributed as dist

from train_system.runtime.plugin import BaseParallelPlugin


class Zero1Plugin(BaseParallelPlugin):

    owns_optimizer = True

    def __init__(self, dp_group: dist.ProcessGroup, optimizer_cls=torch.optim.AdamW, **optimizer_kwargs):
        self.dp_group = dp_group
        self.world_size = dist.get_world_size(dp_group)
        self.rank = dist.get_rank(dp_group)
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self._buffer = None
        self.per_rank_numel = 0
        self.local_params = None

    def _allocate_buffer(self, model):
        all_params = [p for p in model.parameters() if p.requires_grad]
        total_numel = sum(p.numel() for p in all_params)
        self.per_rank_numel = (total_numel + self.world_size - 1) // self.world_size
        dtype, device = all_params[0].dtype, all_params[0].device
        self._buffer = torch.zeros(self.per_rank_numel * self.world_size, dtype=dtype, device=device)

    def _copy_to_buffer(self, model, from_grad=False):
        all_params = [p for p in model.parameters() if p.requires_grad]
        offset = 0
        for p in all_params:
            to_copy = p.grad if from_grad else p.data
            self._buffer[offset:offset+p.numel()].copy_(to_copy.view(-1))
            offset += p.numel()

    def _update_params(self, model: nn.Module):
        all_params = [p for p in model.parameters() if p.requires_grad]
        offset = 0
        for p in all_params:
            p.data.copy_(self._buffer[offset:offset+p.numel()].view_as(p))
            offset += p.numel()

    def setup_model(self, model: nn.Module) -> nn.Module:
        self._allocate_buffer(model)
        self._copy_to_buffer(model, from_grad=False)
        self.local_params = nn.Parameter(self._buffer[self.rank*self.per_rank_numel : (self.rank+1)*self.per_rank_numel].clone())
        self.optimizer = self.optimizer_cls([self.local_params], **self.optimizer_kwargs)
        return model

    def after_backward(self, model: nn.Module):
        self._copy_to_buffer(model, from_grad=True)
        dist.all_reduce(self._buffer, op=dist.ReduceOp.AVG, group=self.dp_group)
        self.local_params.grad = self._buffer[self.rank*self.per_rank_numel : (self.rank+1)*self.per_rank_numel]

    def step(self, model: nn.Module):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        
        dist.all_gather_into_tensor(self._buffer, self.local_params.data.contiguous(), group=self.dp_group)
        self._update_params(model)

