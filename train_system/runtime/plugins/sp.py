from absl import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from train_system.runtime.plugin import BaseParallelPlugin, ParallelizableModule


class SpPlugin(BaseParallelPlugin):

    def __init__(self, sp_group: dist.ProcessGroup):
        self.sp_group = sp_group
        self.rank = dist.get_rank(sp_group)
        self.world_size = dist.get_world_size(sp_group)

    def _make_all_gather_hook(self, comm_dim):
        def hook(module, input):
            input, *args = input
            buffers = [torch.empty_like(input) for _ in range(self.world_size)]
            dist.all_gather(buffers, input, group=self.sp_group)
            input = torch.cat(buffers, dim=comm_dim)
            return (input, *args)
        return hook

    def _make_scatter_hook(self, comm_dim):
        def hook(module, input, output):
            per_rank_dim = output.size(comm_dim) // self.world_size
            output = torch.narrow(output, dim=comm_dim, start=self.rank*per_rank_dim, length=per_rank_dim)
            return output
        return hook
    
    def setup_model(self, model: nn.Module) -> nn.Module:
        if not isinstance(model, ParallelizableModule):
            logging.warn(f"model {model._get_name()} did not impl parallelize_spec, ignoring SpPlugin!")
            return model
        spec = model.parallelize_spec()
        for r in spec.rules:
            if r.shard_style != "seq":
                continue
            mod = model.get_submodule(r.module_path)
            if r.pre_comm == "all_gather":
                mod.register_forward_pre_hook(self._make_all_gather_hook(r.comm_dim))
            elif r.post_comm == "scatter":
                mod.register_forward_hook(self._make_scatter_hook(r.comm_dim))
            else:
                print(f"Sp plugin only support shard_style=seq, pre_comm=all_gather or post_comm=scatter, skip pre_comm={r.pre_comm}")
        return model