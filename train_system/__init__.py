"""Training system skeleton with composable parallel runtime."""

from .parallel.plan import ParallelPlan, ProcessMesh, MeshAxis
from .runtime.context import RuntimeContext
from .engine.trainer import Trainer

__all__ = ["ParallelPlan", "ProcessMesh", "MeshAxis", "RuntimeContext", "Trainer"]
