from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from data import SimpleTensorDataLoader
from parallel import ParallelPlan
from runtime import MeshConfig, RuntimeCore
from train import Trainer, TrainerConfig
from utils.metrics import MetricAggregator, MetricLogger, MetricReduction, MetricRule


class LossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.proj(batch).pow(2).mean()


class BatchMeanLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.scale * batch.mean()


class CaptureLogger(MetricLogger):
    def __init__(self) -> None:
        self.records: list[dict[str, float | int | str | bool | None]] = []

    def log(self, metrics: dict[str, float | int | str | bool | None]) -> None:
        self.records.append(dict(metrics))


class CaptureCheckpointUploader:
    def __init__(self, every_steps: int) -> None:
        self.every_steps = every_steps
        self.uploads: list[tuple[Path, int]] = []
        self.closed = False

    def upload_checkpoint(self, checkpoint_dir: str | Path, step: int) -> None:
        if step % self.every_steps == 0:
            self.uploads.append((Path(checkpoint_dir), step))

    def close(self) -> None:
        self.closed = True


def _make_runtime(seed: int = 1234, grad_accum_steps: int = 1) -> RuntimeCore:
    torch.manual_seed(seed)
    model = LossModel()
    return RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        grad_accum_steps=grad_accum_steps,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.01),
    )


def _make_loader() -> SimpleTensorDataLoader:
    data = torch.arange(80, dtype=torch.float32).view(20, 4) / 80
    return SimpleTensorDataLoader(data, batch_size=2)


def _make_scalar_loader() -> SimpleTensorDataLoader:
    data = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
    return SimpleTensorDataLoader(data, batch_size=1)


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


def test_trainer_aggregates_metrics_over_log_interval() -> None:
    runtime = _make_runtime()
    logger = CaptureLogger()
    aggregator = MetricAggregator(
        rules={
            "step": MetricRule(MetricReduction.LAST, MetricReduction.RANK0),
            "loss": MetricRule(MetricReduction.MEAN, MetricReduction.RANK0),
        }
    )
    trainer = Trainer(
        runtime=runtime,
        dataloader=_make_loader(),
        config=TrainerConfig(max_steps=4, log_every=2),
        logger=logger,
        metric_aggregator=aggregator,
    )
    trainer.setup()
    expected_losses = []
    while runtime.state.step < 4:
        batch = trainer.dataloader.next_batch()
        _, should_step = runtime.run_step(batch)
        runtime.step_optimizer()
        expected_losses.append(float(runtime.state.loss.detach().float().item()))
    expected_log_losses = [
        sum(expected_losses[0:2]) / 2,
        sum(expected_losses[2:4]) / 2,
    ]

    runtime = _make_runtime()
    logger = CaptureLogger()
    trainer = Trainer(
        runtime=runtime,
        dataloader=_make_loader(),
        config=TrainerConfig(max_steps=4, log_every=2),
        logger=logger,
        metric_aggregator=MetricAggregator(
            rules={
                "step": MetricRule(MetricReduction.LAST, MetricReduction.RANK0),
                "loss": MetricRule(MetricReduction.MEAN, MetricReduction.RANK0),
            }
        ),
    )
    trainer.setup()
    trainer.fit()

    assert [record["step"] for record in logger.records] == [2, 4]
    for actual, expected in zip([record["loss"] for record in logger.records], expected_log_losses):
        assert abs(float(actual) - expected) < 1e-8


def test_trainer_collects_each_microbatch_metric() -> None:
    model = BatchMeanLossModel()
    runtime = RuntimeCore(
        mesh=MeshConfig(),
        plan=ParallelPlan(),
        model=model,
        grad_accum_steps=2,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.0),
    )
    logger = CaptureLogger()
    trainer = Trainer(
        runtime=runtime,
        dataloader=_make_scalar_loader(),
        config=TrainerConfig(max_steps=1, log_every=1),
        logger=logger,
        metric_aggregator=MetricAggregator(
            rules={
                "loss": MetricRule(MetricReduction.MEAN, MetricReduction.RANK0),
                "step": MetricRule(MetricReduction.LAST, MetricReduction.RANK0),
            }
        ),
    )
    trainer.setup()
    trainer.fit()

    assert len(logger.records) == 1
    assert logger.records[0]["step"] == 1
    assert abs(float(logger.records[0]["loss"]) - 1.0) < 1e-8


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
        assert not (checkpoint_dir / "step_00000002.tmp").exists()

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


def test_trainer_checkpoint_uploader_follows_upload_cadence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp) / "ckpts"
        uploader = CaptureCheckpointUploader(every_steps=2)
        runtime = _make_runtime()
        trainer = Trainer(
            runtime=runtime,
            dataloader=_make_loader(),
            config=TrainerConfig(max_steps=4, checkpoint_every=1, checkpoint_dir=checkpoint_dir),
            checkpoint_uploader=uploader,
        )
        trainer.setup()
        trainer.fit()

        assert uploader.closed
        assert [step for _, step in uploader.uploads] == [2, 4]
        assert [path.name for path, _ in uploader.uploads] == ["step_00000002", "step_00000004"]


def test_trainer_checkpoint_retention() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp) / "ckpts"
        runtime = _make_runtime()
        trainer = Trainer(
            runtime=runtime,
            dataloader=_make_loader(),
            config=TrainerConfig(
                max_steps=4,
                checkpoint_every=1,
                checkpoint_dir=checkpoint_dir,
                checkpoint_keep_last=1,
                checkpoint_keep_every_n_steps=2,
            ),
        )
        trainer.setup()
        trainer.fit()

        assert sorted(path.name for path in checkpoint_dir.iterdir()) == ["step_00000002", "step_00000004"]


def test_trainer_checkpoint_min_free_space_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp) / "ckpts"
        runtime = _make_runtime()
        trainer = Trainer(
            runtime=runtime,
            dataloader=_make_loader(),
            config=TrainerConfig(
                max_steps=1,
                checkpoint_every=1,
                checkpoint_dir=checkpoint_dir,
                checkpoint_min_free_gb=1_000_000_000,
            ),
        )
        trainer.setup()
        try:
            trainer.fit()
        except RuntimeError as exc:
            assert "insufficient free space" in str(exc)
        else:
            raise AssertionError("checkpoint free-space guard did not raise")


def test_metric_aggregator_step_reduction() -> None:
    aggregator = MetricAggregator()
    aggregator.update({"step": 1, "perf/step_sec": 3.0, "train/tokens": 10.0, "perf/world_size": 1})
    aggregator.update({"step": 2, "perf/step_sec": 5.0, "train/tokens": 14.0, "perf/world_size": 1})

    metrics = aggregator.flush(step_delta=2)

    assert metrics["perf/step_sec_window"] == 8.0
    assert metrics["perf/step_sec"] == 4.0
    assert metrics["train/tokens"] == 24.0
    assert metrics["train/tokens_per_sec"] == 3.0


def main() -> None:
    test_trainer_logs_optimizer_steps()
    test_trainer_aggregates_metrics_over_log_interval()
    test_trainer_collects_each_microbatch_metric()
    test_trainer_checkpoint_resume_matches_continuous()
    test_trainer_checkpoint_uploader_follows_upload_cadence()
    test_trainer_checkpoint_retention()
    test_trainer_checkpoint_min_free_space_raises()
    test_metric_aggregator_step_reduction()
    print("trainer loop smoke ok")


if __name__ == "__main__":
    main()
