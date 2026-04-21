from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.distributed as dist

from train_system.runtime.plugin import BaseParallelPlugin


@dataclass
class _Bucket:
    params: list[nn.Parameter]
    start: int
    end: int
    shard_start: int
    shard_end: int
    local_params: nn.Parameter
    pending: int
    handle: dist.Work | None = None


class Zero1Plugin(BaseParallelPlugin):

    owns_optimizer = True

    def __init__(self, dp_group: dist.ProcessGroup, bucket_mb_size: int, optimizer_cls=torch.optim.AdamW, **optimizer_kwargs):
        self.dp_group = dp_group
        self.world_size = dist.get_world_size(dp_group)
        self.rank = dist.get_rank(dp_group)
        self.bucket_byte_size = bucket_mb_size * 1024 * 1024
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self._data_buffer = None
        self._grad_buffer = None
        self.buckets = []

    def _reset_buckets(self):
        for bkt in self.buckets:
            bkt.pending = len(bkt.params)
            bkt.handle = None

    def _add_param_hooks(self):
        def make_hook(bkt_inst):
            def hook(p):
                bkt_inst.pending -= 1
                if bkt_inst.pending == 0:
                    if bkt_inst.local_params.grad is None:
                        bkt_inst.local_params.grad = torch.empty_like(bkt_inst.local_params.data)
                    bkt_inst.handle = dist.reduce_scatter_tensor(bkt_inst.local_params.grad, self._grad_buffer[bkt_inst.start:bkt_inst.end], op=dist.ReduceOp.AVG, group=self.dp_group, async_op=True)
            return hook
        for bkt in self.buckets:
            for p in bkt.params:
                p.register_post_accumulate_grad_hook(make_hook(bkt))

    def _prepare_buffer_and_buckets(self, model):
        all_params = [p for p in model.parameters() if p.requires_grad][::-1]
        dtype, device = all_params[0].dtype, all_params[0].device
        param_buckets = []
        curr_bytes = 0
        curr_bucket = []
        # TODO: consider multiply for hardware alignment
        pad_to_size = self.world_size
        def padded_len(l):
            return (l+(pad_to_size-1))//pad_to_size*pad_to_size
        for p in all_params:
            curr_bytes += p.numel() * p.element_size()
            curr_bucket.append(p)
            if curr_bytes >= self.bucket_byte_size:
                param_buckets.append(curr_bucket[:])
                curr_bytes = 0
                curr_bucket = []
        if curr_bytes > 0:
            param_buckets.append(curr_bucket[:])
        param_padded_size = [padded_len(sum(p.numel() for p in bkt)) for bkt in param_buckets]
        self._data_buffer = torch.zeros(sum(param_padded_size), dtype=dtype, device=device)
        self._grad_buffer = torch.zeros(sum(param_padded_size), dtype=dtype, device=device)
        offset = 0
        for params, padded_size in zip(param_buckets, param_padded_size):
            assert padded_size % self.world_size == 0, f"Bucket padded numel({padded_size}) should be divisible by world_size({self.world_size})!"
            param_offset = offset
            for p in params:
                with torch.no_grad():
                    self._data_buffer[param_offset:param_offset+p.numel()].copy_(p.view(-1))
                    p.data = self._data_buffer[param_offset:param_offset+p.numel()].view_as(p)
                    p.grad = self._grad_buffer[param_offset:param_offset+p.numel()].view_as(p)
                param_offset+=p.numel()
            per_rank_size = padded_size // self.world_size
            shard_start = offset+self.rank*per_rank_size
            shard_end = offset+(self.rank+1)*per_rank_size
            self.buckets.append(
                _Bucket(
                    params=params,
                    start=offset,
                    end=offset+padded_size,
                    shard_start=shard_start,
                    shard_end=shard_end,
                    local_params=nn.Parameter(self._data_buffer[shard_start:shard_end].clone()),
                    pending=len(params)
                )
            )
            offset += padded_size
        self._reset_buckets()
        self._add_param_hooks()

    def setup_model(self, model: nn.Module) -> nn.Module:
        self._prepare_buffer_and_buckets(model)
        self.optimizer = self.optimizer_cls([bkt.local_params for bkt in self.buckets], **self.optimizer_kwargs)
        return model

    def before_forward(self, model: nn.Module) -> None:
        self._reset_buckets()

    def after_backward(self, model: nn.Module) -> None:
        for bkt in self.buckets:
            assert bkt.handle is not None, "Bucket handle cannot be None after backward!"
            bkt.handle.wait()

    def step(self, model: nn.Module):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._grad_buffer.zero_()
        with torch.no_grad():
            for bkt in self.buckets:
                bkt.handle = dist.all_gather_into_tensor(self._data_buffer[bkt.start:bkt.end], bkt.local_params.data.contiguous(), group=self.dp_group, async_op=True)
            for bkt in self.buckets:
                bkt.handle.wait()

