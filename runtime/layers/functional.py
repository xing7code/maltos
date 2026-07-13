import torch
import torch.distributed as dist
import torch.nn.functional as F

from runtime.buffer_allocator import allocate_buffer
from utils.distributed import all_gather_single, reduce_scatter_single


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        group: dist.ProcessGroup,
        comm_dim: int,
        alloc_key: str,
        backward_reduce_op: dist.ReduceOp | None,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.comm_dim = comm_dim
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        ctx.backward_reduce_op = backward_reduce_op

        x_t = x.transpose(0, comm_dim).contiguous()
        out_shape = (ctx.world_size * x_t.shape[0], *x_t.shape[1:])
        out_t = allocate_buffer(
            key=f"{alloc_key}.forward",
            shape=out_shape,
            dtype=x.dtype,
            device=x.device,
        )
        all_gather_single(out_t, x_t, group=group)
        return out_t.transpose(0, comm_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dim = ctx.comm_dim
        per_rank_dim = grad_output.shape[dim] // ctx.world_size
        grad = grad_output.narrow(dim, ctx.rank * per_rank_dim, per_rank_dim).contiguous()
        if ctx.backward_reduce_op is not None:
            dist.all_reduce(grad, op=ctx.backward_reduce_op, group=ctx.group)
        return grad, None, None, None, None


class ReduceScatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup, comm_dim: int, reduce_op: dist.ReduceOp, alloc_key: str) -> torch.Tensor:
        ctx.group = group
        ctx.comm_dim = comm_dim
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        ctx.alloc_key = alloc_key

        x_t = x.transpose(0, comm_dim).contiguous()
        shape = list(x_t.size())
        shape[0] //= ctx.world_size
        out = allocate_buffer(
            key=f"{alloc_key}.forward",
            shape=tuple(shape),
            dtype=x.dtype,
            device=x.device,
        )
        reduce_scatter_single(out, x_t, group=ctx.group, op=reduce_op)
        return out.transpose(0, comm_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output_t = grad_output.transpose(0, ctx.comm_dim).contiguous()
        out_shape = (ctx.world_size * grad_output_t.shape[0], *grad_output_t.shape[1:])
        out_t = allocate_buffer(
            key=f"{ctx.alloc_key}.backward",
            shape=out_shape,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        all_gather_single(out_t, grad_output_t, group=ctx.group)
        return out_t.transpose(0, ctx.comm_dim).contiguous(), None, None, None, None


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup, reduce_op: dist.ReduceOp) -> torch.Tensor:
        ctx.group = group
        ctx.reduce_op = reduce_op
        out = x.contiguous().clone()
        dist.all_reduce(out, op=reduce_op, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.contiguous(), None, None


class _RowParallelReduceScatterAsync(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, tp_group, alloc_key: str):
        ctx.tp_group = tp_group
        ctx.alloc_key = alloc_key
        ctx.use_bias = bias is not None
        ctx.input_shape = tuple(input.shape)
        ctx.rank = dist.get_rank(tp_group)
        ctx.world_size = dist.get_world_size(tp_group)
        ctx.save_for_backward(input, weight)

        output = F.linear(input, weight, None)
        output = reduce_scatter(
            output,
            tp_group,
            1,
            alloc_key=f"{alloc_key}.forward.reduce_scatter",
        )
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_output_weight = weight.to(dtype=grad_output.dtype)
        grad_weight_dtype = weight.dtype

        grad_output_t = grad_output.transpose(0, 1).contiguous()
        gathered_shape = (ctx.world_size * grad_output_t.shape[0], *grad_output_t.shape[1:])
        gathered_grad_output_t = allocate_buffer(
            key=f"{ctx.alloc_key}.backward.all_gather",
            shape=gathered_shape,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        handle = all_gather_single(
            gathered_grad_output_t,
            grad_output_t,
            group=ctx.tp_group,
            async_op=True,
        )

        grad_input = torch.empty(ctx.input_shape, dtype=grad_output.dtype, device=grad_output.device)
        local_seq = grad_output.shape[1]
        local_start = ctx.rank * local_seq
        local_end = local_start + local_seq
        grad_input.narrow(1, local_start, local_seq).copy_(grad_output.matmul(grad_output_weight))

        handle.wait()

        gathered_grad_output = gathered_grad_output_t.transpose(0, 1)
        if local_start > 0:
            left = gathered_grad_output.narrow(1, 0, local_start)
            grad_input.narrow(1, 0, local_start).copy_(left.matmul(grad_output_weight))
        if local_end < grad_input.shape[1]:
            right = gathered_grad_output.narrow(1, local_end, grad_input.shape[1] - local_end)
            grad_input.narrow(1, local_end, grad_input.shape[1] - local_end).copy_(right.matmul(grad_output_weight))

        grad_weight = gathered_grad_output.reshape(-1, gathered_grad_output.shape[-1]).to(grad_weight_dtype).t().matmul(
            input.reshape(-1, input.shape[-1]).to(grad_weight_dtype)
        )
        grad_bias = gathered_grad_output.to(grad_weight_dtype).sum(dim=(0, 1)) if ctx.use_bias else None
        return grad_input, grad_weight, grad_bias, None, None


class _RingShift(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        group: dist.ProcessGroup,
        send_to: int,
        recv_from: int,
        alloc_key: str,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.send_to = send_to
        ctx.recv_from = recv_from
        ctx.alloc_key = alloc_key
        if dist.get_world_size(group) == 1:
            return x
        out = allocate_buffer(
            key=f"{alloc_key}.forward",
            shape=tuple(x.shape),
            dtype=x.dtype,
            device=x.device,
        )
        send_global_rank = dist.get_global_rank(group, send_to)
        recv_global_rank = dist.get_global_rank(group, recv_from)
        _pairwise_send_recv(
            x.contiguous(),
            out,
            send_rank=send_global_rank,
            recv_rank=recv_global_rank,
            group=group,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if dist.get_world_size(ctx.group) == 1:
            return grad_output, None, None, None, None
        grad_input = allocate_buffer(
            key=f"{ctx.alloc_key}.backward",
            shape=tuple(grad_output.shape),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        send_global_rank = dist.get_global_rank(ctx.group, ctx.recv_from)
        recv_global_rank = dist.get_global_rank(ctx.group, ctx.send_to)
        _pairwise_send_recv(
            grad_output.contiguous(),
            grad_input,
            send_rank=send_global_rank,
            recv_rank=recv_global_rank,
            group=ctx.group,
        )
        return grad_input, None, None, None, None


def all_gather(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    comm_dim: int = 0,
    *,
    alloc_key: str = "layers.functional.all_gather",
    backward_reduce_op: dist.ReduceOp | None = None,
) -> torch.Tensor:
    return AllGather.apply(x, group, comm_dim, alloc_key, backward_reduce_op)

def reduce_scatter(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    comm_dim: int = 0,
    reduce_op: dist.ReduceOp = dist.ReduceOp.SUM,
    *,
    alloc_key: str = "layers.functional.reduce_scatter",
) -> torch.Tensor:
    return ReduceScatter.apply(x, group, comm_dim, reduce_op, alloc_key)

def all_reduce(x: torch.Tensor, group: dist.ProcessGroup, reduce_op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
    return AllReduce.apply(x, group, reduce_op)


def row_parallel_reduce_scatter_async(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    tp_group: dist.ProcessGroup,
    *,
    alloc_key: str,
) -> torch.Tensor:
    return _RowParallelReduceScatterAsync.apply(input, weight, bias, tp_group, alloc_key)


def ring_shift(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    send_to: int,
    recv_from: int,
    *,
    alloc_key: str,
) -> torch.Tensor:
    return _RingShift.apply(x, group, send_to, recv_from, alloc_key)


def _pairwise_send_recv(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    *,
    send_rank: int,
    recv_rank: int,
    group: dist.ProcessGroup,
) -> None:
    ops = [
        dist.P2POp(dist.isend, send_tensor, send_rank, group),
        dist.P2POp(dist.irecv, recv_tensor, recv_rank, group),
    ]
    for work in dist.batch_isend_irecv(ops):
        work.wait()
