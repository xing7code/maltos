from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from train_system.runtime.plugin import MetricValue


class MetricLogger(Protocol):
    def log(self, metrics: dict[str, MetricValue]) -> None: ...


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


def _format_metrics(metrics: dict[str, MetricValue]) -> str:
    parts = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.6g}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)
