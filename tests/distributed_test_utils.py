from __future__ import annotations

from collections.abc import Callable
import os
import socket
from typing import Any, Protocol

import torch
import torch.distributed as dist

from parallel.specs import TpSpShardAxis
from runtime import MeshAxis, RuntimeCore
from runtime.plugins.ep import _ExpertParallelMoE


def _configure_default_gloo_socket_ifname() -> None:
    if "GLOO_SOCKET_IFNAME" in os.environ:
        return
    for _, ifname in socket.if_nameindex():
        if ifname == "lo" or ifname.startswith("lo"):
            os.environ.setdefault("GLOO_SOCKET_IFNAME", ifname)
            return


_configure_default_gloo_socket_ifname()


def configure_process_group_env() -> None:
    _configure_default_gloo_socket_ifname()


class _TpSpSpecModel(Protocol):
    def tpsp_parallelize_spec(self): ...


def _identity_name(name: str) -> str:
    return name


def supports_bf16_autocast() -> bool:
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            _ = torch.randn(8, 8) @ torch.randn(8, 8)
        return True
    except Exception:
        return False


def normalize_param_name(name: str) -> str:
    return name[len("module.") :] if name.startswith("module.") else name


def all_gather_tensor(
    tensor: torch.Tensor,
    group: dist.ProcessGroup | None,
) -> list[torch.Tensor]:
    if group is None:
        return [tensor.detach().clone()]
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return gathered


def rule_by_param_name(model: _TpSpSpecModel | Any) -> dict[str, str]:
    if not hasattr(model, "tpsp_parallelize_spec"):
        return {}
    rules = {}
    for rule in model.tpsp_parallelize_spec().rules:
        if rule.shard_axis in (TpSpShardAxis.PARAM_OUT, TpSpShardAxis.PARAM_IN):
            rules[f"{rule.module_path}.weight"] = rule.shard_axis
            rules[f"{rule.module_path}.bias"] = rule.shard_axis
    return rules


def logical_tensor(
    name: str,
    tensor: torch.Tensor,
    shard_rules: dict[str, str],
    tp_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    shard_axis = shard_rules.get(name)
    if shard_axis == TpSpShardAxis.PARAM_OUT:
        return torch.cat(all_gather_tensor(tensor, tp_group), dim=0)
    if shard_axis == TpSpShardAxis.PARAM_IN:
        return torch.cat(all_gather_tensor(tensor, tp_group), dim=1)
    return tensor.detach().clone()


def max_diff(lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]) -> tuple[str, float]:
    worst_name = ""
    worst_diff = 0.0
    for name, lhs_tensor in lhs.items():
        rhs_tensor = rhs[name].to(lhs_tensor.device, lhs_tensor.dtype)
        diff = (lhs_tensor - rhs_tensor).abs().max().item()
        if diff > worst_diff:
            worst_name = name
            worst_diff = diff
    return worst_name, worst_diff


def reduce_loss(loss: torch.Tensor, core: RuntimeCore) -> torch.Tensor:
    reduced = loss.detach().clone()
    if (cp_group := core.get_group(MeshAxis.CP)) is not None:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=cp_group)
    if (dp_group := core.get_group(MeshAxis.DP)) is not None:
        dist.all_reduce(reduced, op=dist.ReduceOp.AVG, group=dp_group)
    return reduced


def named_tensors(
    model: torch.nn.Module,
    shard_rules: dict[str, str],
    tp_group: dist.ProcessGroup | None,
    *,
    grads: bool = False,
    normalize_name: Callable[[str], str] = _identity_name,
    device: torch.device | str | None = None,
    skip_local_experts: bool = False,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if skip_local_experts and ".local_experts." in name:
            continue
        logical_name = normalize_name(name)
        source = param.grad if grads else param.detach()
        if source is None:
            source = torch.zeros_like(param)
        tensor = logical_tensor(logical_name, source.detach(), shard_rules, tp_group)
        if device is not None:
            tensor = tensor.to(device)
        tensors[logical_name] = tensor
    return tensors


def bucket_grad_by_param_id(
    zero_plugin: Any,
) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for bucket in zero_plugin.buckets:
        if bucket.local_param.grad is None:
            continue
        shard_group = bucket.group_context.group
        if shard_group is not None:
            full_grad = torch.cat(all_gather_tensor(bucket.local_param.grad.detach(), shard_group), dim=0)
        else:
            full_grad = bucket.local_param.grad.detach().clone()
        param_shapes = getattr(bucket, "param_shapes", None)
        param_numels = getattr(bucket, "param_numels", None)
        if param_shapes is not None and param_numels is not None:
            offset = 0
            for param, numel, shape in zip(bucket.params, param_numels, param_shapes):
                result[id(param)] = full_grad[offset : offset + numel].view(shape).cpu()
                offset += numel
            continue
        offset = 0
        for param in bucket.params:
            numel = param.numel()
            result[id(param)] = full_grad[offset : offset + numel].view(param.shape).cpu()
            offset += numel
    return result


def named_zero_bucket_grads(
    model: torch.nn.Module,
    zero_plugin: Any,
    shard_rules: dict[str, str],
    tp_group: dist.ProcessGroup | None,
    *,
    normalize_name: Callable[[str], str] = _identity_name,
    device: torch.device | str | None = None,
    skip_local_experts: bool = False,
) -> dict[str, torch.Tensor]:
    grad_by_pid = bucket_grad_by_param_id(zero_plugin)
    tensors: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if skip_local_experts and ".local_experts." in name:
            continue
        logical_name = normalize_name(name)
        grad = grad_by_pid.get(id(param), torch.zeros_like(param).cpu())
        tensor = logical_tensor(logical_name, grad.to(param.device), shard_rules, tp_group)
        if device is not None:
            tensor = tensor.to(device)
        tensors[logical_name] = tensor
    return tensors


def gather_object_dict(
    local: dict[str, torch.Tensor],
    group: dist.ProcessGroup | None,
) -> dict[str, torch.Tensor]:
    if group is None:
        return local
    gathered: list[dict[str, torch.Tensor]] = [None for _ in range(dist.get_world_size(group))]  # type: ignore[list-item]
    dist.all_gather_object(gathered, local, group=group)
    merged: dict[str, torch.Tensor] = {}
    for shard in gathered:
        merged.update(shard)
    return merged


def moe_local_expert_tensors(
    model: torch.nn.Module,
    *,
    grads: bool = False,
    grad_by_pid: dict[int, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, _ExpertParallelMoE):
            continue
        for local_idx, global_idx in enumerate(module.local_expert_ids):
            expert = module.local_experts[local_idx]
            for param_name, param in expert.named_parameters():
                full_name = f"{module_name}.experts.{global_idx}.{param_name}"
                if grad_by_pid is not None:
                    tensor = grad_by_pid.get(id(param), torch.zeros_like(param).cpu())
                else:
                    source = param.grad if grads else param.detach()
                    if source is None:
                        source = torch.zeros_like(param)
                    tensor = source.detach().clone().cpu()
                tensors[full_name] = tensor
    return tensors


def moe_named_tensors(
    model: torch.nn.Module,
    shard_rules: dict[str, str],
    tp_group: dist.ProcessGroup | None,
    ep_group: dist.ProcessGroup | None,
    *,
    pp_group: dist.ProcessGroup | None = None,
    grads: bool = False,
    pp_partitioned: bool = False,
    normalize_name: Callable[[str], str] = _identity_name,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    output_device = torch.device(device) if device is not None else next(model.parameters()).device
    shared_tensors = named_tensors(
        model,
        shard_rules,
        tp_group,
        grads=grads,
        normalize_name=normalize_name,
        device=torch.device("cpu"),
        skip_local_experts=True,
    )
    if pp_partitioned:
        shared_tensors = gather_object_dict(shared_tensors, pp_group)
    tensors = {name: tensor.to(output_device) for name, tensor in shared_tensors.items()}
    expert_tensors = gather_object_dict(moe_local_expert_tensors(model, grads=grads), ep_group)
    if pp_partitioned:
        expert_tensors = gather_object_dict(expert_tensors, pp_group)
    for name, tensor in expert_tensors.items():
        tensors[name] = tensor.to(output_device)
    return tensors


def moe_zero_named_grads(
    model: torch.nn.Module,
    zero_plugin: Any,
    shard_rules: dict[str, str],
    tp_group: dist.ProcessGroup | None,
    ep_group: dist.ProcessGroup | None,
    *,
    pp_group: dist.ProcessGroup | None = None,
    pp_partitioned: bool = False,
    normalize_name: Callable[[str], str] = _identity_name,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    output_device = torch.device(device) if device is not None else next(model.parameters()).device
    grad_by_pid = bucket_grad_by_param_id(zero_plugin)
    shared_tensors = named_zero_bucket_grads(
        model,
        zero_plugin,
        shard_rules,
        tp_group,
        normalize_name=normalize_name,
        device=torch.device("cpu"),
        skip_local_experts=True,
    )
    if pp_partitioned:
        shared_tensors = gather_object_dict(shared_tensors, pp_group)
    tensors = {name: tensor.to(output_device) for name, tensor in shared_tensors.items()}
    expert_tensors = gather_object_dict(moe_local_expert_tensors(model, grad_by_pid=grad_by_pid), ep_group)
    if pp_partitioned:
        expert_tensors = gather_object_dict(expert_tensors, pp_group)
    for name, tensor in expert_tensors.items():
        tensors[name] = tensor.to(output_device)
    return tensors
