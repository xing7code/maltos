from __future__ import annotations

from typing import Any, Protocol


class DataLoaderStateProtocol(Protocol):
    consumed_tokens: int


class StatefulDataLoaderProtocol(Protocol):
    def next_batch(self) -> Any: ...

    def state_dict(self) -> DataLoaderStateProtocol: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...
