from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch.distributed as dist

from train_system.data.protocols import StatefulDataLoaderProtocol
from train_system.runtime.core import RuntimeCore
from train_system.state.checkpoint import load_sharded_checkpoint, save_sharded_checkpoint
from train_system.utils.metrics import MetricAggregator, MetricLogger


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
        logger: MetricLogger | None = None,
        metric_aggregator: MetricAggregator | None = None,
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
        self.logger = logger
        self.metric_aggregator = metric_aggregator or MetricAggregator()

    def setup(self) -> None:
        self.runtime.setup()
        self.runtime.state_manager.bind_dataloader(self.dataloader)
        if self.config.resume_from is not None:
            load_sharded_checkpoint(self.runtime.state_manager, self.config.resume_from)

    def fit(self) -> None:
        while self.runtime.state.step < self.config.max_steps:
            previous_step = self.runtime.state.step
            batch = self.dataloader.next_batch()
            self.runtime.run_train_step(batch)
            self.metric_aggregator.update(self.runtime.collect_metrics())
            if self.runtime.state.step == previous_step:
                continue
            self._maybe_log()
            self._maybe_checkpoint()

    def _maybe_log(self) -> None:
        step = self.runtime.state.step
        if step == 0 or step % self.config.log_every != 0:
            return
        metrics = self.metric_aggregator.flush()
        if self.logger is None:
            return
        if not _is_log_rank():
            return
        self.logger.log(metrics)

    def _maybe_checkpoint(self) -> None:
        if self.config.checkpoint_every is None:
            return
        step = self.runtime.state.step
        if step == 0 or step % self.config.checkpoint_every != 0:
            return
        assert self.config.checkpoint_dir is not None
        save_sharded_checkpoint(self.runtime.state_manager, _checkpoint_step_dir(self.config.checkpoint_dir, step))


def _checkpoint_step_dir(checkpoint_dir: str | Path, step: int) -> Path:
    return Path(checkpoint_dir) / f"step_{step:08d}"


def _is_log_rank() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0
