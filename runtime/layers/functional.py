from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from runtime.buffer_allocator import allocate_buffer
from runtime.layers.flash_utils import flash_attn_dense_backward, flash_attn_dense_with_lse
from utils.distributed import all_gather_single, reduce_scatter_single


@dataclass
class _AsyncRingExchange:
    send_tensor: torch.Tensor
    recv_tensor: torch.Tensor
    works: list[dist.Work]

    def wait(self) -> torch.Tensor:
        for work in self.works:
            work.wait()
        return self.recv_tensor


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


def flash_ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    module_id: int,
    mb_idx: int,
) -> torch.Tensor:
    return _FlashRingAttentionFunc.apply(q, k, v, group, module_id, mb_idx)


def _pairwise_send_recv(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    *,
    send_rank: int,
    recv_rank: int,
    group: dist.ProcessGroup,
) -> None:
    for work in _pairwise_send_recv_async(
        send_tensor,
        recv_tensor,
        send_rank=send_rank,
        recv_rank=recv_rank,
        group=group,
    ):
        work.wait()


def _pairwise_send_recv_async(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    *,
    send_rank: int,
    recv_rank: int,
    group: dist.ProcessGroup,
) -> list[dist.Work]:
    ops = [
        dist.P2POp(dist.isend, send_tensor, send_rank, group),
        dist.P2POp(dist.irecv, recv_tensor, recv_rank, group),
    ]
    return list(dist.batch_isend_irecv(ops))


def _ring_exchange_tensor_async(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    send_to: int,
    recv_from: int,
    *,
    alloc_key: str,
) -> _AsyncRingExchange:
    if dist.get_world_size(group) == 1:
        return _AsyncRingExchange(send_tensor=x, recv_tensor=x, works=[])
    send_tensor = x.contiguous()
    recv_tensor = allocate_buffer(
        key=f"{alloc_key}.async",
        shape=tuple(x.shape),
        dtype=x.dtype,
        device=x.device,
    )
    send_global_rank = dist.get_global_rank(group, send_to)
    recv_global_rank = dist.get_global_rank(group, recv_from)
    works = _pairwise_send_recv_async(
        send_tensor,
        recv_tensor,
        send_rank=send_global_rank,
        recv_rank=recv_global_rank,
        group=group,
    )
    return _AsyncRingExchange(
        send_tensor=send_tensor,
        recv_tensor=recv_tensor,
        works=works,
    )


def _merge_attention_blocks(
    running_out: torch.Tensor,
    running_lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    *,
    q_slice: slice | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q_slice is not None:
        current_out = running_out[:, :, q_slice, :]
        current_lse = running_lse[:, :, q_slice]
        new_lse = torch.logaddexp(current_lse, block_lse)
        prev_scale = _exp_delta_or_zero(current_lse, new_lse)
        block_scale = _exp_delta_or_zero(block_lse, new_lse)
        running_out[:, :, q_slice, :] = (
            current_out * prev_scale.unsqueeze(-1) + block_out.float() * block_scale.unsqueeze(-1)
        )
        running_lse[:, :, q_slice] = new_lse
        return running_out, running_lse
    new_lse = torch.logaddexp(running_lse, block_lse)
    prev_scale = _exp_delta_or_zero(running_lse, new_lse)
    block_scale = _exp_delta_or_zero(block_lse, new_lse)
    new_out = running_out * prev_scale.unsqueeze(-1) + block_out.float() * block_scale.unsqueeze(-1)
    return new_out, new_lse


def _exp_delta_or_zero(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(target)
    finite = torch.isfinite(source)
    if finite.any():
        out[finite] = torch.exp(source[finite] - target[finite])
    return out


class _FlashRingAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        group: dist.ProcessGroup,
        module_id: int,
        mb_idx: int,
    ) -> torch.Tensor:
        out, softmax_lse = _flash_ring_forward(
            q,
            k,
            v,
            group=group,
            module_id=module_id,
            mb_idx=mb_idx,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.group = group
        ctx.module_id = module_id
        ctx.mb_idx = mb_idx
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = _flash_ring_backward(
            dout.contiguous(),
            q,
            k,
            v,
            out,
            softmax_lse,
            group=ctx.group,
            module_id=ctx.module_id,
            mb_idx=ctx.mb_idx,
        )
        return dq, dk, dv, None, None, None


def _flash_ring_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    module_id: int,
    mb_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    send_to = (rank + 1) % world_size
    recv_from = (rank - 1 + world_size) % world_size
    local_seq_len = q.size(-2)
    half_seq_len = local_seq_len // 2
    q_tail = q[:, :, half_seq_len:, :]
    current_kv = torch.cat([k, v], dim=-1)
    running_out = torch.zeros_like(v, dtype=torch.float32)
    running_lse = torch.full(q.shape[:-1], float("-inf"), dtype=torch.float32, device=q.device)
    for step in range(world_size):
        next_kv_exchange = None
        if step + 1 != world_size:
            next_kv_exchange = _ring_exchange_tensor_async(
                current_kv,
                group,
                send_to,
                recv_from,
                alloc_key=f"cp.flash_ring.{module_id}.mb{mb_idx}.fwd.step_{step}.kv",
            )
        current_k, current_v = current_kv.split(k.size(-1), dim=-1)
        if step == 0:
            block = flash_attn_dense_with_lse(q, current_k, current_v, causal=True)
            running_out, running_lse = _merge_attention_blocks(running_out, running_lse, block.out, block.lse)
        elif step <= rank:
            block = flash_attn_dense_with_lse(
                q,
                current_k[:, :, :half_seq_len, :],
                current_v[:, :, :half_seq_len, :],
                causal=False,
            )
            running_out, running_lse = _merge_attention_blocks(running_out, running_lse, block.out, block.lse)
        else:
            block = flash_attn_dense_with_lse(q_tail, current_k, current_v, causal=False)
            running_out, running_lse = _merge_attention_blocks(
                running_out,
                running_lse,
                block.out,
                block.lse,
                q_slice=slice(half_seq_len, None),
            )
        if step + 1 == world_size:
            break
        current_kv = next_kv_exchange.wait()
    return running_out.to(dtype=v.dtype), running_lse


def _flash_ring_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    module_id: int,
    mb_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return flash_attn_dense_backward(dout, q, k, v, out, softmax_lse, causal=True)

    rank = dist.get_rank(group)
    send_to = (rank + 1) % world_size
    recv_from = (rank - 1 + world_size) % world_size
    local_seq_len = q.size(-2)
    half_seq_len = local_seq_len // 2
    q_tail = q[:, :, half_seq_len:, :]
    dout_tail = dout[:, :, half_seq_len:, :]
    out_tail = out[:, :, half_seq_len:, :]
    softmax_lse_tail = softmax_lse[:, :, half_seq_len:]
    current_kv = torch.cat([k, v], dim=-1)
    current_dkv = torch.zeros_like(current_kv, dtype=torch.float32)
    dq = torch.zeros_like(q, dtype=torch.float32)
    pending_dkv_exchange: _AsyncRingExchange | None = None

    for step in range(world_size):
        next_kv_exchange = None
        if step + 1 != world_size:
            next_kv_exchange = _ring_exchange_tensor_async(
                current_kv,
                group,
                send_to,
                recv_from,
                alloc_key=f"cp.flash_ring.{module_id}.mb{mb_idx}.bwd.step_{step}.kv",
            )
        current_k, current_v = current_kv.split(k.size(-1), dim=-1)
        if pending_dkv_exchange is not None:
            current_dkv = pending_dkv_exchange.wait()
        if step == 0:
            block_dq, block_dk, block_dv = flash_attn_dense_backward(
                dout,
                q,
                current_k,
                current_v,
                out,
                softmax_lse,
                causal=True,
            )
            dq += block_dq.float()
            current_dkv += torch.cat([block_dk, block_dv], dim=-1).float()
        elif step <= rank:
            block_dq, block_dk, block_dv = flash_attn_dense_backward(
                dout,
                q,
                current_k[:, :, :half_seq_len, :],
                current_v[:, :, :half_seq_len, :],
                out,
                softmax_lse,
                causal=False,
            )
            dq += block_dq.float()
            current_dkv[:, :, :half_seq_len, :] += torch.cat([block_dk, block_dv], dim=-1).float()
        else:
            block_dq, block_dk, block_dv = flash_attn_dense_backward(
                dout_tail,
                q_tail,
                current_k,
                current_v,
                out_tail,
                softmax_lse_tail,
                causal=False,
            )
            dq[:, :, half_seq_len:, :] += block_dq.float()
            current_dkv += torch.cat([block_dk, block_dv], dim=-1).float()
        if step + 1 == world_size:
            break
        pending_dkv_exchange = _ring_exchange_tensor_async(
            current_dkv,
            group,
            send_to,
            recv_from,
            alloc_key=f"cp.flash_ring.{module_id}.mb{mb_idx}.bwd.step_{step}.grad",
        )
        current_kv = next_kv_exchange.wait()

    local_dkv = ring_shift(
        current_dkv,
        group,
        send_to,
        recv_from,
        alloc_key=f"cp.flash_ring.{module_id}.mb{mb_idx}.bwd.return",
    )
    local_dk, local_dv = local_dkv.split(k.size(-1), dim=-1)
    return dq.to(dtype=q.dtype), local_dk.to(dtype=k.dtype), local_dv.to(dtype=v.dtype)
