from __future__ import annotations

import torch
import torch.distributed as dist


def all_gather_single(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
):
    fn = getattr(dist, "all_gather_single", None)
    if fn is not None:
        return fn(output_tensor, input_tensor, group=group, async_op=async_op)
    return dist.all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=async_op)


def reduce_scatter_single(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    *,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
):
    fn = getattr(dist, "reduce_scatter_single", None)
    if fn is not None:
        return fn(output_tensor, input_tensor, op=op, group=group, async_op=async_op)
    return dist.reduce_scatter_tensor(output_tensor, input_tensor, op=op, group=group, async_op=async_op)


def distributed_barrier() -> None:
    if not dist.is_initialized():
        return
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
        return
    dist.barrier()
