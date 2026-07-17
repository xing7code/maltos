from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist


_GLOO_LOW_PRECISION_DTYPES = {
    torch.float16,
    torch.bfloat16,
}


def _copy_cast(dst: torch.Tensor, src: torch.Tensor) -> None:
    with torch.no_grad():
        dst.copy_(src.to(dtype=dst.dtype))


class CompletedWork:
    def wait(self) -> None:
        pass

    def block_current_stream(self) -> None:
        pass


class PostProcessWork:
    def __init__(
        self,
        works: list[dist.Work],
        *,
        post_wait: Callable[[], None] | None = None,
        keepalive: list[torch.Tensor] | None = None,
    ) -> None:
        self.works = works
        self.post_wait = post_wait
        self.keepalive = keepalive or []
        self._done = False

    def wait(self) -> None:
        if self._done:
            return
        for work in self.works:
            work.wait()
        if self.post_wait is not None:
            self.post_wait()
        self._done = True

    def block_current_stream(self) -> None:
        if self._done:
            return
        if self.post_wait is not None:
            self.wait()
            return
        for work in self.works:
            work.block_current_stream()


def is_gloo_group(group: dist.ProcessGroup | None = None) -> bool:
    if not dist.is_initialized():
        return False
    backend = dist.get_backend(group) if group is not None else dist.get_backend()
    return backend == "gloo"


def needs_fp32_comm(tensor: torch.Tensor, group: dist.ProcessGroup | None = None) -> bool:
    return is_gloo_group(group) and tensor.dtype in _GLOO_LOW_PRECISION_DTYPES


def all_gather_single(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
):
    if needs_fp32_comm(input_tensor, group):
        gathered = torch.empty_like(output_tensor, dtype=torch.float32)
        local = input_tensor.float().contiguous()
        fn = getattr(dist, "all_gather_single", None)
        if fn is not None:
            work = fn(gathered, local, group=group, async_op=True)
        else:
            work = dist.all_gather_into_tensor(gathered, local, group=group, async_op=True)
        wrapped = PostProcessWork(
            [work],
            post_wait=lambda: _copy_cast(output_tensor, gathered),
            keepalive=[local, gathered],
        )
        if async_op:
            return wrapped
        wrapped.wait()
        return None

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
    if needs_fp32_comm(input_tensor, group):
        scattered = torch.empty_like(output_tensor, dtype=torch.float32)
        gathered = input_tensor.float().contiguous()
        fn = getattr(dist, "reduce_scatter_single", None)
        if fn is not None:
            work = fn(scattered, gathered, op=op, group=group, async_op=True)
        else:
            work = dist.reduce_scatter_tensor(scattered, gathered, op=op, group=group, async_op=True)
        wrapped = PostProcessWork(
            [work],
            post_wait=lambda: _copy_cast(output_tensor, scattered),
            keepalive=[gathered, scattered],
        )
        if async_op:
            return wrapped
        wrapped.wait()
        return None

    fn = getattr(dist, "reduce_scatter_single", None)
    if fn is not None:
        return fn(output_tensor, input_tensor, op=op, group=group, async_op=async_op)
    return dist.reduce_scatter_tensor(output_tensor, input_tensor, op=op, group=group, async_op=async_op)


def all_reduce_tensor(
    tensor: torch.Tensor,
    *,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
):
    if needs_fp32_comm(tensor, group):
        reduced = tensor.float().contiguous()
        work = dist.all_reduce(reduced, op=op, group=group, async_op=True)
        wrapped = PostProcessWork(
            [work],
            post_wait=lambda: _copy_cast(tensor, reduced),
            keepalive=[reduced],
        )
        if async_op:
            return wrapped
        wrapped.wait()
        return None
    return dist.all_reduce(tensor, op=op, group=group, async_op=async_op)


def isend_tensor_async(
    tensor: torch.Tensor,
    dst: int,
    *,
    group: dist.ProcessGroup | None = None,
):
    if needs_fp32_comm(tensor, group):
        send_tensor = tensor.float().contiguous()
        works = list(dist.batch_isend_irecv([dist.P2POp(dist.isend, send_tensor, dst, group)]))
        return PostProcessWork(works, keepalive=[send_tensor])
    works = list(dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor, dst, group)]))
    return works[0]


def irecv_tensor_async(
    tensor: torch.Tensor,
    src: int,
    *,
    group: dist.ProcessGroup | None = None,
):
    if needs_fp32_comm(tensor, group):
        recv_tensor = torch.empty_like(tensor, dtype=torch.float32)
        works = list(dist.batch_isend_irecv([dist.P2POp(dist.irecv, recv_tensor, src, group)]))
        return PostProcessWork(
            works,
            post_wait=lambda: _copy_cast(tensor, recv_tensor),
            keepalive=[recv_tensor],
        )
    works = list(dist.batch_isend_irecv([dist.P2POp(dist.irecv, tensor, src, group)]))
    return works[0]


def pairwise_send_recv_async(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    *,
    send_rank: int,
    recv_rank: int,
    group: dist.ProcessGroup,
) -> list[dist.Work | PostProcessWork]:
    if needs_fp32_comm(send_tensor, group) or needs_fp32_comm(recv_tensor, group):
        send_fp32 = send_tensor.float().contiguous()
        recv_fp32 = torch.empty_like(recv_tensor, dtype=torch.float32)
        works = list(
            dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, send_fp32, send_rank, group),
                    dist.P2POp(dist.irecv, recv_fp32, recv_rank, group),
                ]
            )
        )
        return [
            PostProcessWork(
                works,
                post_wait=lambda: _copy_cast(recv_tensor, recv_fp32),
                keepalive=[send_fp32, recv_fp32],
            )
        ]

    ops = [
        dist.P2POp(dist.isend, send_tensor, send_rank, group),
        dist.P2POp(dist.irecv, recv_tensor, recv_rank, group),
    ]
    return list(dist.batch_isend_irecv(ops))


def distributed_barrier() -> None:
    if not dist.is_initialized():
        return
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
        return
    dist.barrier()
