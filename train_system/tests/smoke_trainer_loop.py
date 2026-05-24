from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from train_system.data import SimpleTensorDataLoader
from train_system.parallel import ParallelPlan
from train_system.runtime import MeshConfig, RuntimeCore
from train_system.train import Trainer, TrainerConfig
from train_system.utils.metrics import MetricLogger


class LossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.proj(batch).pow(2).mean()


class CaptureLogger(MetricLogger):
    def __init__(self) -> None:
        self.records: list[dict[str, float | int | str | bool | None]] = []

    def log(self, metrics: dict[str, float | int | str | bool | None]) -> None:
        self.records.append(dict(metrics))


def _make_runtime(seed: int = 1234) -> RuntimeCore:
    torch.manual_seed(seed)
    model = LossModel()
    return RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
    )


def _make_loader() -> SimpleTensorDataLoader:
    data = torch.arange(80, dtype=torch.float32).view(20, 4) / 80
    return SimpleTensorDataLoader(data, batch_size=2)


def _params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def test_trainer_logs_optimizer_steps() -> None:
    runtime = _make_runtime()
    logger = CaptureLogger()
    trainer = Trainer(
        runtime=runtime,
        dataloader=_make_loader(),
        config=TrainerConfig(max_steps=3, log_every=1),
        logger=logger,
    )
    trainer.setup()
    trainer.fit()

    assert runtime.state.step == 3
    assert [record["step"] for record in logger.records] == [1, 2, 3]
    assert all("loss" in record for record in logger.records)
    assert all(record["lr"] == 0.01 for record in logger.records)


def test_trainer_checkpoint_resume_matches_continuous() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp) / "ckpts"

        continuous = _make_runtime()
        continuous_trainer = Trainer(
            runtime=continuous,
            dataloader=_make_loader(),
            config=TrainerConfig(max_steps=3),
        )
        continuous_trainer.setup()
        continuous_trainer.fit()

        interrupted = _make_runtime()
        interrupted_trainer = Trainer(
            runtime=interrupted,
            dataloader=_make_loader(),
            config=TrainerConfig(max_steps=2, checkpoint_every=2, checkpoint_dir=checkpoint_dir),
        )
        interrupted_trainer.setup()
        interrupted_trainer.fit()

        restored = _make_runtime(seed=9999)
        restored_trainer = Trainer(
            runtime=restored,
            dataloader=_make_loader(),
            config=TrainerConfig(max_steps=3, resume_from=checkpoint_dir / "step_00000002"),
        )
        restored_trainer.setup()
        assert restored.state.step == 2
        restored_trainer.fit()

        assert restored.state.step == continuous.state.step
        assert restored.state_manager.export_trainer_state().dataloader == continuous.state_manager.export_trainer_state().dataloader
        for name, expected in _params(continuous.model).items():
            actual = _params(restored.model)[name]
            if not torch.allclose(actual, expected, atol=1e-6, rtol=0):
                raise AssertionError(f"param mismatch for {name}: max_diff={(actual - expected).abs().max().item()}")


def main() -> None:
    test_trainer_logs_optimizer_steps()
    test_trainer_checkpoint_resume_matches_continuous()
    print("trainer loop smoke ok")


if __name__ == "__main__":
    main()
