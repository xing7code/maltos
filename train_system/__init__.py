"""Training system skeleton with composable parallel runtime."""

from .parallel.plan import ParallelPlan, ParallelConfig
from .runtime.context import RuntimeContext
from .engine.trainer import Trainer

__all__ = ["ParallelPlan", "ParallelConfig", "RuntimeContext", "Trainer"]
