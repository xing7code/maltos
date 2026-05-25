from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch.distributed as dist

from data.protocols import StatefulDataLoaderProtocol
from runtime.core import RuntimeCore
from state.checkpoint import load_sharded_checkpoint, save_sharded_checkpoint
from utils.metrics import MetricAggregator, MetricLogger


class CheckpointUploader(Protocol):
    def upload_checkpoint(self, checkpoint_dir: str | Path, step: int) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class TrainerConfig:
    max_steps: int
    log_every: int = 1
    checkpoint_every: int | None = None
    checkpoint_dir: str | Path | None = None
    resume_from: str | Path | None = None


class Trainer:
    def __init__(
        self,
        *,
        runtime: RuntimeCore,
        dataloader: StatefulDataLoaderProtocol,
        config: TrainerConfig,
        logger: MetricLogger | list[MetricLogger] | None = None,
        metric_aggregator: MetricAggregator | None = None,
        checkpoint_uploader: CheckpointUploader | None = None,
    ) -> None:
        if config.max_steps < 0:
            raise ValueError(f"max_steps must be >= 0, got {config.max_steps}")
        if config.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {config.log_every}")
        if config.checkpoint_every is not None and config.checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {config.checkpoint_every}")
        if config.checkpoint_every is not None and config.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required when checkpoint_every is set")
        self.runtime = runtime
        self.dataloader = dataloader
        self.config = config
        self.loggers = _normalize_loggers(logger)
        self.metric_aggregator = metric_aggregator or MetricAggregator()
        self.checkpoint_uploader = checkpoint_uploader

    def setup(self) -> None:
        self.runtime.setup()
        self.runtime.state_manager.bind_dataloader(self.dataloader)
        if self.config.resume_from is not None:
            load_sharded_checkpoint(self.runtime.state_manager, self.config.resume_from)

    def fit(self) -> None:
        try:
            while self.runtime.state.step < self.config.max_steps:
                previous_step = self.runtime.state.step
                batch = self.dataloader.next_batch()
                self.runtime.run_train_step(batch)
                metrics = self.runtime.collect_metrics()
                self.metric_aggregator.update(metrics)
                if self.runtime.state.step == previous_step:
                    continue
                self._maybe_log()
                self._maybe_checkpoint()
        finally:
            if self.checkpoint_uploader is not None:
                self.checkpoint_uploader.close()

    def _maybe_log(self) -> None:
        step = self.runtime.state.step
        if step == 0 or step % self.config.log_every != 0:
            return
        metrics = self.metric_aggregator.flush()
        if not self.loggers:
            return
        if not _is_log_rank():
            return
        for logger in self.loggers:
            logger.log(metrics)

    def _maybe_checkpoint(self) -> None:
        if self.config.checkpoint_every is None:
            return
        step = self.runtime.state.step
        if step == 0 or step % self.config.checkpoint_every != 0:
            return
        assert self.config.checkpoint_dir is not None
        checkpoint_dir = _checkpoint_step_dir(self.config.checkpoint_dir, step)
        save_sharded_checkpoint(self.runtime.state_manager, checkpoint_dir)
        if self.checkpoint_uploader is not None and _is_log_rank():
            self.checkpoint_uploader.upload_checkpoint(checkpoint_dir, step)


def _checkpoint_step_dir(checkpoint_dir: str | Path, step: int) -> Path:
    return Path(checkpoint_dir) / f"step_{step:08d}"


def _is_log_rank() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _normalize_loggers(logger: MetricLogger | list[MetricLogger] | None) -> list[MetricLogger]:
    if logger is None:
        return []
    if isinstance(logger, list):
        return logger
    return [logger]
