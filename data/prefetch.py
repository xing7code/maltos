from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from typing import Any

import torch


class PrefetchDataLoader:
    """One-batch CPU prefetch wrapper that preserves checkpoint cursor semantics."""

    def __init__(self, loader) -> None:
        self.loader = loader
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="maltos-data")
        self._future: Future[Any] | None = None
        self._state_before_prefetch: dict[str, Any] | None = None

    def next_batch(self) -> Any:
        if self._future is None:
            batch = self.loader.next_batch()
        else:
            batch = self._future.result()
            self._future = None
            self._state_before_prefetch = None
        self._start_prefetch()
        return _pin_memory(batch)

    def state_dict(self):
        # The background read has advanced the wrapped loader by one batch. A
        # checkpoint must instead record the state after the last *consumed*
        # batch, otherwise resume would silently skip the prefetched batch.
        if self._state_before_prefetch is not None:
            return dict(self._state_before_prefetch)
        return self.loader.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._cancel_prefetch()
        self.loader.load_state_dict(state)

    def close(self) -> None:
        self._cancel_prefetch()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _start_prefetch(self) -> None:
        self._state_before_prefetch = _state_as_dict(self.loader.state_dict())
        self._future = self._executor.submit(self.loader.next_batch)

    def _cancel_prefetch(self) -> None:
        if self._future is not None:
            self._future.cancel()
            self._future = None
        self._state_before_prefetch = None


def _state_as_dict(state: Any) -> dict[str, Any]:
    if is_dataclass(state) and not isinstance(state, type):
        return asdict(state)
    if isinstance(state, dict):
        return dict(state)
    raise TypeError(f"dataloader state must be a dataclass or dict, got {type(state)!r}")


def _pin_memory(value: Any) -> Any:
    if not torch.cuda.is_available():
        return value
    if torch.is_tensor(value) and value.device.type == "cpu":
        return value.pin_memory()
    if isinstance(value, dict):
        return {key: _pin_memory(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_pin_memory(item) for item in value)
    if isinstance(value, list):
        return [_pin_memory(item) for item in value]
    return value
