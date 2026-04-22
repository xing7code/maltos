from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist

from train_system.runtime.plugin import BaseParallelPlugin


@dataclass
class _Bucket:
    module: nn.Module
    params: list[nn.Parameter]
    param_shapes: list[torch.Size]
    param_numels: list[int]
    shard_start: int
    shard_end: int
    buffer_size: int
    local_params: nn.Parameter
    pending: int
    attached: bool = False
    grad_handle: dist.Work | None = None
    fwd_handle: dist.Work | None = None
    bwd_handle: dist.Work | None = None
    data_buffer: torch.Tensor | None = None
    grad_buffer: torch.Tensor | None = None
    prev_bucket: _Bucket | None = None
    next_bucket: _Bucket | None = None

    def make_forward_pre_hook(self):
        def hook(module, input):
            if self.fwd_handle is not None:
                self.fwd_handle.wait()
            offset = 0
            for p, numel, shape in zip(self.params, self.param_numels, self.param_shapes):
                p.data = self.data_buffer[offset:offset+numel].view(shape)
                offset += numel
        return hook

    def gather_data_buffer_fwd(self, group):
        self.fwd_handle = dist.all_gather_into_tensor(self.data_buffer[:self.buffer_size], self.local_params.data, group, async_op=True)

    def gather_data_buffer_bwd(self, group):
        self.bwd_handle = dist.all_gather_into_tensor(self.data_buffer[:self.buffer_size], self.local_params.data, group, async_op=True)

    def make_forward_hook(self, group):
        def hook(module, input, output):
            # freeup data buffer
            for p in self.params:
                p.data = self.local_params.data
            if self.next_bucket is not None:
                self.next_bucket.gather_data_buffer_fwd(group)
        return hook

    def make_backward_pre_hook(self):
        def hook(module, grad_input):
            if self.bwd_handle is not None:
                self.bwd_handle.wait()
            offset = 0
            for p, numel, shape in zip(self.params, self.param_numels, self.param_shapes):
                p.data = self.data_buffer[offset:offset+numel].view(shape)
                offset += numel
        return hook

    def make_backward_hook(self, group):
        def hook(module, grad_input, grad_output):
            # freeup data buffer
            for p in self.params:
                p.data = self.local_params.data
            if self.prev_bucket is not None:
                self.prev_bucket.gather_data_buffer_bwd(group)
        return hook


class Zero3Plugin(BaseParallelPlugin):

    owns_optimizer = True

    def __init__(self, dp_group: dist.ProcessGroup, wrap_cls: set, optimizer_cls=torch.optim.AdamW, **optimizer_kwargs):
        self.dp_group = dp_group
        self.world_size = dist.get_world_size(dp_group)
        self.rank = dist.get_rank(dp_group)
        self.wrap_cls = wrap_cls
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.data_buffer =  [] # two buffers alternative for prefetch
        self.grad_buffer = None
        self.buckets = []

    def _reset_buckets(self):
        for bkt in self.buckets:
            bkt.pending = len(bkt.params)
            bkt.grad_handle = None
            bkt.fwd_handle = None
            bkt.bwd_handle = None
            bkt.attached = False

    def _add_param_hooks(self):
        def make_pre_hook(bkt_inst, next_bkt):
            def hook(grad):
                if not bkt_inst.attached:
                    # grad compute backward, wait for next bkt grad sync.
                    if next_bkt is not None:
                        next_bkt.grad_handle.wait()
                    self.grad_buffer.zero_()
                    offset = 0
                    for p, numel, shape in zip(bkt_inst.params, bkt_inst.param_numels, bkt_inst.param_shapes):
                        p.grad = self.grad_buffer[offset:offset+numel].view(shape)
                        offset += numel
                    bkt_inst.attached = True
                return grad
            return hook
        def make_hook(bkt_inst):
            def hook(p):
                bkt_inst.pending -= 1
                if bkt_inst.pending == 0:
                    if bkt_inst.local_params.grad is None:
                        bkt_inst.local_params.grad = torch.empty_like(bkt_inst.local_params.data)
                    bkt_inst.grad_handle = dist.reduce_scatter_tensor(bkt_inst.local_params.grad, self.grad_buffer[:bkt_inst.buffer_size], op=dist.ReduceOp.AVG, group=self.dp_group, async_op=True)
            return hook
        next_bkt = None
        for bkt in reversed(self.buckets):
            bkt.module.register_forward_pre_hook(bkt.make_forward_pre_hook())
            bkt.module.register_forward_hook(bkt.make_forward_hook(self.dp_group))
            bkt.module.register_full_backward_pre_hook(bkt.make_backward_pre_hook())
            bkt.module.register_full_backward_hook(bkt.make_backward_hook(self.dp_group))
            for p in bkt.params:
                p.register_post_accumulate_grad_hook(make_hook(bkt))
                p.register_hook(make_pre_hook(bkt, next_bkt))
            next_bkt = bkt

    def _prepare_buffer_and_buckets(self, model):
        visited = set()
        # TODO: consider multiply for hardware alignment
        pad_to_size = self.world_size
        def padded_len(l):
            return (l+(pad_to_size-1))//pad_to_size*pad_to_size
        dtype, device = None, None
        for p in model.parameters():
            if p.requires_grad:
                dtype, device=p.dtype, p.device
                break
        max_size = None
        for name, module in model.named_modules():
            if isinstance(module, tuple(self.wrap_cls)):
                if any(name.startswith(v + ".") for v in visited):
                    continue
                visited.add(name)
                params = [p for p in module.parameters() if p.requires_grad]
                if params:
                    padded_size = padded_len(sum(p.numel() for p in params))
                    if max_size is None or max_size<padded_size:
                        max_size = padded_size
                    per_rank_size = padded_size // self.world_size

                    full_param = torch.cat([p.view(-1) for p in params])
                    padded = torch.zeros(padded_size, dtype=dtype, device=device)
                    padded[:full_param.numel()].copy_(full_param)
                    local_data = padded[self.rank*per_rank_size:(self.rank+1)*per_rank_size].clone()
                    local_params=nn.Parameter(local_data)
                    self.buckets.append(
                        _Bucket(
                            module,
                            params,
                            param_shapes=[p.size() for p in params],
                            param_numels=[p.numel() for p in params],
                            shard_start=self.rank*per_rank_size,
                            shard_end=(self.rank+1)*per_rank_size,
                            buffer_size=padded_size,
                            local_params=local_params,
                            pending=len(params)
                        )
                    )
        self.data_buffer = [
            torch.zeros(max_size, dtype=dtype, device=device),
            torch.zeros(max_size, dtype=dtype, device=device),
        ]
        self.grad_buffer = torch.zeros(max_size, dtype=dtype, device=device)
        for i, bkt in enumerate(self.buckets):
            bkt.data_buffer = self.data_buffer[i%2]
            bkt.grad_buffer = self.grad_buffer
            # For alternate prefetch
            if i>=2:
                bkt.prev_bucket = self.buckets[i-2]
            if i<=len(self.buckets)-3:
                bkt.next_bucket = self.buckets[i+2]
        self._reset_buckets()
        self._add_param_hooks()

    def setup_model(self, model: nn.Module) -> nn.Module:
        self._prepare_buffer_and_buckets(model)
        self.optimizer = self.optimizer_cls([bkt.local_params for bkt in self.buckets], **self.optimizer_kwargs)
        return model

    def before_forward(self, model: nn.Module) -> None:
        self._reset_buckets()
        self.buckets[0].gather_data_buffer_fwd(self.dp_group)
        if len(self.buckets) > 1:
            self.buckets[1].gather_data_buffer_fwd(self.dp_group)

    def after_backward(self, model: nn.Module) -> None:
        # previous handles already wait in hooks, only need wait for the last one.
        self.buckets[0].grad_handle.wait()

    def step(self, model: nn.Module):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
