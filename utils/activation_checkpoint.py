from __future__ import annotations

from contextlib import nullcontext
from contextvars import ContextVar, Token


_IS_RECOMPUTING = ContextVar("maltos_activation_checkpoint_recomputing", default=False)


class _RecomputeContext:
    """Marks the second, autograd-driven forward of a non-reentrant checkpoint."""

    def __init__(self) -> None:
        self._token: Token[bool] | None = None

    def __enter__(self) -> "_RecomputeContext":
        self._token = _IS_RECOMPUTING.set(True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self._token is not None
        _IS_RECOMPUTING.reset(self._token)
        self._token = None


def activation_checkpoint_context_fn():
    """Return forward/recompute contexts for ``torch.utils.checkpoint.checkpoint``."""
    return nullcontext(), _RecomputeContext()


def is_activation_checkpoint_recompute() -> bool:
    return _IS_RECOMPUTING.get()
