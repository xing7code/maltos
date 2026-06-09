from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class GroupContext:
    group: dist.ProcessGroup | None
    world_size: int
    rank: int


class AllReduceShardWork:
    def __init__(
        self,
        work: dist.Work,
        grad_buffer: torch.Tensor,
        local_grad: torch.Tensor,
        shard_start: int,
        shard_end: int,
    ):
        self.work = work
        self.grad_buffer = grad_buffer
        self.local_grad = local_grad
        self.shard_start = shard_start
        self.shard_end = shard_end

    def wait(self) -> None:
        self.work.wait()
        self.local_grad.add_(self.grad_buffer[self.shard_start : self.shard_end])


class ReduceScatterShardWork:
    def __init__(
        self,
        work: dist.Work,
        shard_buffer: torch.Tensor,
        local_grad: torch.Tensor,
    ):
        self.work = work
        self.shard_buffer = shard_buffer
        self.local_grad = local_grad

    def wait(self) -> None:
        self.work.wait()
        self.local_grad.add_(self.shard_buffer)
