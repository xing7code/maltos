from __future__ import annotations

import concurrent.futures
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.distributed as dist

from runtime.plugin import MetricValue


class MetricLogger(Protocol):
    def log(self, metrics: dict[str, MetricValue]) -> None: ...


class MetricReduction(str, Enum):
    MEAN = "mean"
    SUM = "sum"
    LAST = "last"
    MAX = "max"
    ANY = "any"
    RANK0 = "rank0"
    NONE = "none"


@dataclass(frozen=True)
class MetricRule:
    local: MetricReduction = MetricReduction.LAST
    distributed: MetricReduction = MetricReduction.RANK0


class MetricAggregator:
    def __init__(
        self,
        rules: dict[str, MetricRule] | None = None,
        syncer: "DistributedMetricSync | None" = None,
    ) -> None:
        self.rules = rules or {}
        self.syncer = syncer or DistributedMetricSync()
        self._values: dict[str, list[MetricValue]] = {}

    def update(self, metrics: dict[str, MetricValue]) -> None:
        for key, value in metrics.items():
            self._values.setdefault(key, []).append(value)

    def flush(self) -> dict[str, MetricValue]:
        local_metrics = {
            key: _reduce_local(values, _rule_for_key(key, self.rules).local)
            for key, values in self._values.items()
        }
        self._values.clear()
        metrics = {
            key: self.syncer.reduce_value(value, _rule_for_key(key, self.rules).distributed)
            for key, value in local_metrics.items()
        }
        _add_derived_metrics(metrics)
        return metrics

    def has_values(self) -> bool:
        return bool(self._values)


class DistributedMetricSync:
    def reduce_value(self, value: MetricValue, reduction: MetricReduction) -> MetricValue:
        if reduction in {MetricReduction.NONE, MetricReduction.RANK0} or not dist.is_initialized():
            return value
        if value is None:
            return None
        if reduction == MetricReduction.ANY:
            tensor = torch.tensor(int(bool(value)), dtype=torch.int64)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            return bool(tensor.item())
        if not isinstance(value, (float, int, bool)):
            return value
        tensor = torch.tensor(float(value), dtype=torch.float64)
        if reduction == MetricReduction.MEAN:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= dist.get_world_size()
            return float(tensor.item())
        if reduction == MetricReduction.SUM:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return float(tensor.item())
        if reduction == MetricReduction.MAX:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            return float(tensor.item())
        if reduction == MetricReduction.LAST:
            return value
        raise ValueError(f"unsupported distributed metric reduction={reduction.value}")


class ConsoleMetricLogger:
    def log(self, metrics: dict[str, MetricValue]) -> None:
        print(_format_metrics(metrics))


class JsonlMetricLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict[str, MetricValue]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")


class WandbMetricLogger:
    def __init__(
        self,
        *,
        project: str,
        name: str | None = None,
        entity: str | None = None,
        mode: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        import wandb

        self.wandb = wandb
        self.run = wandb.init(
            project=project,
            name=name,
            entity=entity,
            mode=mode,
            tags=tags,
            config=config,
        )

    def log(self, metrics: dict[str, MetricValue]) -> None:
        step = metrics.get("step")
        self.wandb.log(metrics, step=int(step) if isinstance(step, int) else None)


class WandbCheckpointUploader:
    def __init__(
        self,
        logger: WandbMetricLogger,
        *,
        every_steps: int,
        artifact_prefix: str | None = None,
    ) -> None:
        if every_steps < 1:
            raise ValueError(f"every_steps must be >= 1, got {every_steps}")
        self.logger = logger
        self.every_steps = every_steps
        self.artifact_prefix = _sanitize_artifact_name(artifact_prefix or logger.run.name or logger.run.id)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._future: concurrent.futures.Future[None] | None = None

    def upload_checkpoint(self, checkpoint_dir: str | Path, step: int) -> None:
        if step % self.every_steps != 0:
            return
        self._check_previous_upload()
        if self._future is not None and not self._future.done():
            print(f"warning: previous W&B checkpoint upload still running; skip step={step}")
            return
        path = Path(checkpoint_dir)
        self._future = self.executor.submit(self._upload, path, step)

    def close(self) -> None:
        self._check_previous_upload(wait=True)
        self.executor.shutdown(wait=True)

    def _check_previous_upload(self, wait: bool = False) -> None:
        if self._future is None:
            return
        if wait:
            concurrent.futures.wait([self._future])
        if self._future.done():
            exc = self._future.exception()
            if exc is not None:
                print(f"warning: W&B checkpoint upload failed: {exc}")

    def _upload(self, checkpoint_dir: Path, step: int) -> None:
        artifact = self.logger.wandb.Artifact(
            name=f"{self.artifact_prefix}-step-{step:08d}",
            type="checkpoint",
            metadata={"step": step, "path": str(checkpoint_dir)},
        )
        artifact.add_dir(str(checkpoint_dir))
        self.logger.run.log_artifact(
            artifact,
            aliases=["latest", f"step_{step}"],
        )


def _format_metrics(metrics: dict[str, MetricValue]) -> str:
    parts = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.6g}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _sanitize_artifact_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-._")
    return sanitized or "checkpoint"


def _rule_for_key(key: str, rules: dict[str, MetricRule]) -> MetricRule:
    if key in rules:
        return rules[key]
    if key == "loss" or key.endswith("/loss"):
        return MetricRule(MetricReduction.MEAN, MetricReduction.MEAN)
    if key == "lr" or key.endswith("/lr"):
        return MetricRule(MetricReduction.LAST, MetricReduction.RANK0)
    if key in {"step", "microbatch_idx", "grad_accum_steps", "scheduler_last_epoch"}:
        return MetricRule(MetricReduction.LAST, MetricReduction.RANK0)
    if key == "should_step_optimizer":
        return MetricRule(MetricReduction.LAST, MetricReduction.RANK0)
    if key.endswith("_sec"):
        return MetricRule(MetricReduction.SUM, MetricReduction.MAX)
    if key.endswith("/overflow"):
        return MetricRule(MetricReduction.ANY, MetricReduction.ANY)
    if key.endswith("/grad_norm") or "memory" in key:
        return MetricRule(MetricReduction.MAX, MetricReduction.MAX)
    if "tokens" in key:
        return MetricRule(MetricReduction.SUM, MetricReduction.SUM)
    return MetricRule(MetricReduction.LAST, MetricReduction.RANK0)


def _reduce_local(values: list[MetricValue], reduction: MetricReduction) -> MetricValue:
    if not values:
        return None
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    if reduction in {MetricReduction.LAST, MetricReduction.RANK0, MetricReduction.NONE}:
        return filtered[-1]
    if reduction == MetricReduction.ANY:
        return any(bool(value) for value in filtered)
    if not all(isinstance(value, (float, int, bool)) for value in filtered):
        return filtered[-1]
    numeric_values = [float(value) for value in filtered]
    if reduction == MetricReduction.MEAN:
        return sum(numeric_values) / len(numeric_values)
    if reduction == MetricReduction.SUM:
        return sum(numeric_values)
    if reduction == MetricReduction.MAX:
        return max(numeric_values)
    raise ValueError(f"unsupported local metric reduction={reduction.value}")


def _add_derived_metrics(metrics: dict[str, MetricValue]) -> None:
    tokens = metrics.get("train/tokens")
    elapsed_sec = metrics.get("perf/step_sec")
    if isinstance(tokens, (float, int)) and isinstance(elapsed_sec, (float, int)) and elapsed_sec > 0:
        metrics["train/tokens_per_sec"] = float(tokens) / float(elapsed_sec)
    tokens_per_sec = metrics.get("train/tokens_per_sec")
    flops_per_token = metrics.get("perf/flops_per_token")
    world_size = metrics.get("perf/world_size")
    if (
        isinstance(tokens_per_sec, (float, int))
        and isinstance(flops_per_token, (float, int))
        and isinstance(world_size, (float, int))
        and world_size > 0
    ):
        metrics["perf/tflops_per_gpu"] = float(tokens_per_sec) * float(flops_per_token) / float(world_size) / 1e12
