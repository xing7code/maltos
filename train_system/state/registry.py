from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from train_system.state.param_handle import ParamHandle, ParamRuntimeMetadata, ParamShardMetadata


@dataclass
class StateRegistry:
    """Runtime-owned index of logical training state.

    The current implementation only tracks parameters, but the registry is the
    place where ZeRO/FSDP, optimizer sharding, checkpointing, and profilers will
    agree on logical parameter names, local shards, materialization state, and
    eventually gradient and optimizer-state handles.
    """

    params: dict[str, ParamHandle] = field(default_factory=dict)

    def register_module(self, model: nn.Module) -> None:
        self.params.clear()
        for fq_name, param in model.named_parameters():
            self.params[fq_name] = ParamHandle(
                param=param,
                shard=ParamShardMetadata(
                    fq_name=fq_name,
                    numel=param.numel(),
                    shard_offset=0,
                    shard_numel=param.numel(),
                    owner_rank=0,
                ),
                runtime=ParamRuntimeMetadata(),
            )

    def get_param(self, fq_name: str) -> ParamHandle:
        return self.params[fq_name]

    def items(self):
        return self.params.items()
