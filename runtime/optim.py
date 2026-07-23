from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn


class MasterWeightsOptimizer(torch.optim.Optimizer):
    """Optimizer wrapper that keeps model params in low precision but updates fp32 masters."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        optimizer_factory,
        master_dtype: torch.dtype = torch.float32,
    ) -> None:
        model_params = [param for param in params if param.requires_grad]
        if not model_params:
            raise ValueError("MasterWeightsOptimizer requires at least one trainable parameter")
        self.model_params = model_params
        self.master_dtype = master_dtype
        self.master_params = [
            nn.Parameter(param.detach().to(dtype=master_dtype), requires_grad=True)
            for param in self.model_params
        ]
        self._model_param_by_master_id = {
            id(master_param): model_param
            for model_param, master_param in zip(self.model_params, self.master_params, strict=True)
        }
        super().__init__(self.master_params, defaults={})
        self._optimizer = optimizer_factory(self.master_params)
        self.param_groups = self._optimizer.param_groups
        self.defaults = self._optimizer.defaults
        self.state = self._optimizer.state
        self._master_grads_copied = False

    def model_param_for(self, param: nn.Parameter) -> nn.Parameter:
        return self._model_param_by_master_id.get(id(param), param)

    def copy_master_params(self) -> None:
        if self._master_grads_copied:
            return
        for model_param, master_param in zip(self.model_params, self.master_params, strict=True):
            grad = model_param.grad
            if grad is None:
                master_param.grad = None
                continue
            master_grad = grad.detach().to(device=master_param.device, dtype=master_param.dtype)
            if master_param.grad is None:
                master_param.grad = master_grad.clone()
            else:
                master_param.grad.copy_(master_grad)
        self._master_grads_copied = True

    def sync_master_params_from_model(self) -> None:
        """Refresh FP32 masters after model parameters were loaded externally."""
        with torch.no_grad():
            for model_param, master_param in zip(self.model_params, self.master_params, strict=True):
                master_param.copy_(model_param.to(device=master_param.device, dtype=master_param.dtype))
        self._master_grads_copied = False

    def step(self, closure=None):
        self.copy_master_params()
        loss = self._optimizer.step(closure)
        with torch.no_grad():
            for model_param, master_param in zip(self.model_params, self.master_params, strict=True):
                model_param.copy_(master_param.to(dtype=model_param.dtype))
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optimizer.zero_grad(set_to_none=set_to_none)
        for model_param in self.model_params:
            if set_to_none:
                model_param.grad = None
            elif model_param.grad is not None:
                model_param.grad.zero_()
        self._master_grads_copied = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "inner_optimizer": self._optimizer.state_dict(),
            "master_params": [param.detach().cpu().clone() for param in self.master_params],
            "master_dtype": str(self.master_dtype),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        inner_state = state_dict.get("inner_optimizer", state_dict)
        self._optimizer.load_state_dict(inner_state)
        self.param_groups = self._optimizer.param_groups
        self.defaults = self._optimizer.defaults
        self.state = self._optimizer.state
        master_params = state_dict.get("master_params")
        if isinstance(master_params, list):
            for master_param, saved in zip(self.master_params, master_params, strict=True):
                if not torch.is_tensor(saved):
                    raise TypeError("master_params entries must be tensors")
                master_param.data.copy_(saved.to(device=master_param.device, dtype=master_param.dtype))
        with torch.no_grad():
            for model_param, master_param in zip(self.model_params, self.master_params, strict=True):
                model_param.copy_(master_param.to(dtype=model_param.dtype))
        self._master_grads_copied = False
