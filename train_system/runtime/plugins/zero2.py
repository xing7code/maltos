# Note: zero2 and zero1 there are no big difference per impl.
# Even zero2 shard all params grad, we still need a buffer to save full grad before sharding.
# The only difference is lifetime of this buffer, zero1 need full lifetime, while zero2 only need it until current bucket.
#
# Though, we implement here as another strategy for both zero1 and zero2:
#     Instead of maintaining global buffer like Zero1Plugin, we move the buffer to per bucket.
#     And this buffer can be reused among different bucket.
#     (There's downside of doing this, is each bucket has to wait for previous bucket reduce_scatter finish.) 

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
    # status if params attach grad to the global buffer.
    attached: bool = False


class Zero2Plugin(BaseParallelPlugin):

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
            bkt.attached = False

    def _add_param_hooks(self):
        def make_hook(bkt_inst):
            def hook(p):
                bkt_inst.pending -= 1
                if bkt_inst.pending == 0:
                    if bkt_inst.local_params.grad is None:
                        bkt_inst.local_params.grad = torch.empty_like(bkt_inst.local_params.data)
                    bkt_inst.handle = dist.reduce_scatter_tensor(bkt_inst.local_params.grad, self._grad_buffer[:bkt_inst.end-bkt_inst.start], op=dist.ReduceOp.AVG, group=self.dp_group, async_op=True)
            return hook
        for bkt in self.buckets:
            for p in bkt.params:
                p.register_post_accumulate_grad_hook(make_hook(bkt))
        def make_pre_hook(bkt_inst, prev_bkt):
            def hook(grad):
                if not bkt_inst.attached:
                    if prev_bkt is not None:
                        prev_bkt.handle.wait()
                    self._grad_buffer.zero_()
                    offset = 0
                    for p in bkt_inst.params:
                        p.grad = self._grad_buffer[offset:offset+p.numel()].view_as(p)
                        offset += p.numel()
                    bkt_inst.attached = True
                return grad
            return hook
        prev_bkt = None
        for bkt in self.buckets:
            for p in bkt.params:
                p.register_hook(make_pre_hook(bkt, prev_bkt))
            prev_bkt = bkt

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
            if curr_bytes + p.numel() * p.element_size() > self.bucket_byte_size:
                param_buckets.append(curr_bucket[:])
                curr_bytes = 0
                curr_bucket = []
            curr_bytes += p.numel() * p.element_size()
            curr_bucket.append(p)
        if curr_bytes > 0:
            param_buckets.append(curr_bucket[:])
        param_padded_size = [padded_len(sum(p.numel() for p in bkt)) for bkt in param_buckets]
        self._data_buffer = torch.zeros(sum(param_padded_size), dtype=dtype, device=device)
        # Different from Zero1Plugin, we set grad_buffer to the biggest bucket, and reuse for all.
        self._grad_buffer = torch.zeros(max(param_padded_size), dtype=dtype, device=device)
        offset = 0
        for params, padded_size in zip(param_buckets, param_padded_size):
            assert padded_size % self.world_size == 0, f"Bucket padded numel({padded_size}) should be divisible by world_size({self.world_size})!"
            param_offset = offset
            for p in params:
                with torch.no_grad():
                    self._data_buffer[param_offset:param_offset+p.numel()].copy_(p.view(-1))
                    p.data = self._data_buffer[param_offset:param_offset+p.numel()].view_as(p)
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
        # previous handles already wait in hooks, only need wait for the last one.
        self.buckets[-1].handle.wait()

    def step(self, model: nn.Module):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for bkt in self.buckets:
                bkt.handle = dist.all_gather_into_tensor(self._data_buffer[bkt.start:bkt.end], bkt.local_params.data.contiguous(), group=self.dp_group, async_op=True)
            for bkt in self.buckets:
                bkt.handle.wait()

