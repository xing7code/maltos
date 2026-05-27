from __future__ import annotations

import torch
import torch.distributed as dist


def distributed_barrier() -> None:
    if not dist.is_initialized():
        return
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
        return
    dist.barrier()
