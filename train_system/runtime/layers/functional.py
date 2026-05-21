import torch
import torch.distributed as dist


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup, comm_dim: int) -> torch.Tensor:
        ctx.group = group
        ctx.comm_dim = comm_dim
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)

        out = [torch.empty_like(x) for _ in range(ctx.world_size)]
        dist.all_gather(out, x.contiguous(), group=group)
        return torch.cat(out, dim=comm_dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dim = ctx.comm_dim
        per_rank_dim = grad_output.shape[dim] // ctx.world_size
        grad = grad_output.narrow(dim, ctx.rank * per_rank_dim, per_rank_dim).contiguous()
        return grad, None, None


class ReduceScatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup, comm_dim: int, reduce_op: dist.ReduceOp) -> torch.Tensor:
        ctx.group = group
        ctx.comm_dim = comm_dim
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)

        x_t = x.transpose(0, comm_dim).contiguous()
        shape = list(x_t.size())
        shape[0] //= ctx.world_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.reduce_scatter_tensor(out, x_t, group=ctx.group, op=reduce_op)
        return out.transpose(0, comm_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        out = [torch.empty_like(grad_output) for _ in range(ctx.world_size)]
        dist.all_gather(out, grad_output.contiguous(), group=ctx.group)
        return torch.cat(out, dim=ctx.comm_dim), None, None, None


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


def all_gather(x: torch.Tensor, group: dist.ProcessGroup, comm_dim: int = 0) -> torch.Tensor:
    return AllGather.apply(x, group, comm_dim)

def reduce_scatter(x: torch.Tensor, group: dist.ProcessGroup, comm_dim: int = 0, reduce_op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
    return ReduceScatter.apply(x, group, comm_dim, reduce_op)

def all_reduce(x: torch.Tensor, group: dist.ProcessGroup, reduce_op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
    return AllReduce.apply(x, group, reduce_op)
