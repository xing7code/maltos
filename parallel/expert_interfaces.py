from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch.nn as nn


@runtime_checkable
class ExpertParallelMoEModule(Protocol):
    @property
    def router(self) -> nn.Module: ...

    @property
    def experts(self) -> nn.ModuleList: ...

    @property
    def num_experts(self) -> int: ...

    @property
    def dim(self) -> int: ...

    @property
    def hidden_size(self) -> int: ...
