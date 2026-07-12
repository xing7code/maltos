from __future__ import annotations

from typing import Protocol, runtime_checkable

from .specs import ContextParallelSpec, ExpertParallelSpec, PipelineParallelSpec, TpSpParallelSpec


@runtime_checkable
class TpSpParallelizableModule(Protocol):
    def tpsp_parallelize_spec(self) -> TpSpParallelSpec: ...


@runtime_checkable
class PipelineParallelizableModule(Protocol):
    def pipeline_parallel_spec(self) -> PipelineParallelSpec: ...


@runtime_checkable
class ContextParallelizableModule(Protocol):
    def context_parallel_spec(self) -> ContextParallelSpec: ...


@runtime_checkable
class ExpertParallelizableModule(Protocol):
    def expert_parallel_spec(self) -> ExpertParallelSpec: ...


@runtime_checkable
class FlopsEstimatableModule(Protocol):
    def flops_per_token(self) -> float: ...
