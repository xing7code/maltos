from __future__ import annotations

import torch

from runtime.optim import MasterWeightsOptimizer


def test_external_model_load_refreshes_master_weights() -> None:
    model_param = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    optimizer = MasterWeightsOptimizer(
        [model_param],
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=1.0),
    )

    # This mirrors a weights-only checkpoint import after optimizer setup.
    with torch.no_grad():
        model_param.fill_(3.0)
    optimizer.sync_master_params_from_model()

    model_param.grad = torch.zeros_like(model_param)
    optimizer.step()
    torch.testing.assert_close(model_param, torch.full_like(model_param, 3.0))


if __name__ == "__main__":
    test_external_model_load_refreshes_master_weights()
    print("PASS")
