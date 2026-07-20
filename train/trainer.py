from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Any, Protocol

import torch.distributed as dist

from data.protocols import StatefulDataLoaderProtocol
from runtime.core import RuntimeCore
from state.checkpoint import save_runtime_spec, save_sharded_checkpoint
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
    load_weights_only: bool = False
    checkpoint_keep_last: int | None = None
    checkpoint_keep_every_n_steps: int | None = None
    checkpoint_min_free_gb: float | None = None


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
        runtime_spec: dict[str, Any] | None = None,
    ) -> None:
        if config.max_steps < 0:
            raise ValueError(f"max_steps must be >= 0, got {config.max_steps}")
        if config.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {config.log_every}")
        if config.checkpoint_every is not None and config.checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {config.checkpoint_every}")
        if config.checkpoint_every is not None and config.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required when checkpoint_every is set")
        if config.checkpoint_every is not None and runtime_spec is None:
            raise ValueError("runtime_spec is required when checkpoint_every is set")
        if config.checkpoint_keep_last is not None and config.checkpoint_keep_last < 0:
            raise ValueError(f"checkpoint_keep_last must be >= 0, got {config.checkpoint_keep_last}")
        if config.checkpoint_keep_every_n_steps is not None and config.checkpoint_keep_every_n_steps < 1:
            raise ValueError(
                "checkpoint_keep_every_n_steps must be >= 1, "
                f"got {config.checkpoint_keep_every_n_steps}"
            )
        if config.checkpoint_min_free_gb is not None and config.checkpoint_min_free_gb < 0:
            raise ValueError(f"checkpoint_min_free_gb must be >= 0, got {config.checkpoint_min_free_gb}")
        self.runtime = runtime
        self.dataloader = dataloader
        self.config = config
        self.loggers = _normalize_loggers(logger)
        self.metric_aggregator = metric_aggregator or MetricAggregator()
        self.checkpoint_uploader = checkpoint_uploader
        self.runtime_spec = runtime_spec
        self._last_logged_step = 0

    def setup(self) -> None:
        self.runtime.state_manager.bind_dataloader(self.dataloader)
        self.runtime.setup(
            checkpoint_path=self.config.resume_from,
            load_weights_only=self.config.load_weights_only,
        )
        if self.config.checkpoint_every is not None:
            assert self.config.checkpoint_dir is not None
            assert self.runtime_spec is not None
            if _is_log_rank():
                save_runtime_spec(self.config.checkpoint_dir, self.runtime_spec)
            if dist.is_initialized():
                dist.barrier()
        self._last_logged_step = self.runtime.state.step_context.step

    def fit(self) -> None:
        try:
            while self.runtime.state.step_context.step < self.config.max_steps:
                batch = self.dataloader.next_batch()
                _, should_step = self.runtime.run_step(batch)
                if should_step:
                    self.runtime.step_optimizer()
                metrics = self.runtime.collect_metrics()
                self.metric_aggregator.update(metrics)
                if not should_step:
                    continue
                self._maybe_log()
                self._maybe_checkpoint()
        finally:
            self.runtime.close()
            if self.checkpoint_uploader is not None:
                self.checkpoint_uploader.close()

    def _maybe_log(self) -> None:
        step = self.runtime.state.step_context.step
        if step == 0 or step % self.config.log_every != 0:
            return
        step_delta = step - self._last_logged_step
        metrics = self.metric_aggregator.flush(step_delta=step_delta)
        self._last_logged_step = step
        if not self.loggers:
            return
        if not _is_log_rank():
            return
        for logger in self.loggers:
            logger.log(metrics)

    def _maybe_checkpoint(self) -> None:
        if self.config.checkpoint_every is None:
            return
        step = self.runtime.state.step_context.step
        if step == 0 or step % self.config.checkpoint_every != 0:
            return
        assert self.config.checkpoint_dir is not None
        checkpoint_dir = _checkpoint_step_dir(self.config.checkpoint_dir, step)
        save_sharded_checkpoint(
            self.runtime.state_manager,
            checkpoint_dir,
            min_free_gb=self.config.checkpoint_min_free_gb,
        )
        _apply_checkpoint_retention(
            self.config.checkpoint_dir,
            current_step=step,
            keep_last=self.config.checkpoint_keep_last,
            keep_every_n_steps=self.config.checkpoint_keep_every_n_steps,
        )
        if self.checkpoint_uploader is not None and _is_log_rank():
            self.checkpoint_uploader.upload_checkpoint(checkpoint_dir, step)


def _checkpoint_step_dir(checkpoint_dir: str | Path, step: int) -> Path:
    return Path(checkpoint_dir) / f"step_{step:08d}"


def _apply_checkpoint_retention(
    checkpoint_dir: str | Path,
    *,
    current_step: int,
    keep_last: int | None,
    keep_every_n_steps: int | None,
) -> None:
    if keep_last is None and keep_every_n_steps is None:
        return
    if not _is_log_rank():
        return
    root = Path(checkpoint_dir)
    step_dirs = sorted(_checkpoint_step_dirs(root), key=lambda item: item[0])
    protected: set[Path] = {path for step, path in step_dirs if step == current_step}
    if keep_last is not None and keep_last > 0:
        protected.update(path for _, path in step_dirs[-keep_last:])
    if keep_every_n_steps is not None:
        protected.update(path for step, path in step_dirs if step % keep_every_n_steps == 0)
    for _, path in step_dirs:
        if path not in protected:
            shutil.rmtree(path)


def _checkpoint_step_dirs(root: Path) -> list[tuple[int, Path]]:
    if not root.is_dir():
        return []
    step_dirs: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"step_(\d{8})", path.name)
        if match is None:
            continue
        step_dirs.append((int(match.group(1)), path))
    return step_dirs


def _is_log_rank() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _normalize_loggers(logger: MetricLogger | list[MetricLogger] | None) -> list[MetricLogger]:
    if logger is None:
        return []
    if isinstance(logger, list):
        return logger
    return [logger]
