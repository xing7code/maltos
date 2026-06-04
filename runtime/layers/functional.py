import torch
import torch.distributed as dist

from runtime.buffer_allocator import allocate_buffer


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

        out = [
            allocate_buffer(
                key=f"{alloc_key}.slot{i}",
                shape=tuple(x.shape),
                dtype=x.dtype,
                device=x.device,
            )
            for i in range(ctx.world_size)
        ]
        dist.all_gather(out, x.contiguous(), group=group)
        return torch.cat(out, dim=comm_dim)

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
        dist.reduce_scatter_tensor(out, x_t, group=ctx.group, op=reduce_op)
        return out.transpose(0, comm_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        out = [
            allocate_buffer(
                key=f"{ctx.alloc_key}.backward.slot{i}",
                shape=tuple(grad_output.shape),
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
            for i in range(ctx.world_size)
        ]
        dist.all_gather(out, grad_output.contiguous(), group=ctx.group)
        return torch.cat(out, dim=ctx.comm_dim), None, None, None, None


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
