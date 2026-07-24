"""Small, framework-wide helpers for semantic PyTorch profiler spans."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import torch


P = ParamSpec("P")
T = TypeVar("T")
ProfileName = str | Callable[P, str]


def profiled(name: ProfileName[P]) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Annotate a method with a fixed or argument-derived profiler span.

    ``name`` may be a string, or a callable receiving the decorated function's
    arguments. The latter keeps repeated operations such as PP actions or
    ZeRO-3 forward/backward materialization distinct in Perfetto.
    """

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            label = name(*args, **kwargs) if callable(name) else name
            with torch.profiler.record_function(label):
                return function(*args, **kwargs)

        return wrapped

    return decorate
