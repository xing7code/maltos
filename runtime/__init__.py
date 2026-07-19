from __future__ import annotations

from importlib import import_module

__all__ = [
    "BufferHandle",
    "BufferPolicy",
    "ContextParallelizableModule",
    "DefaultStepRunner",
    "ExpertParallelizableModule",
    "MeshAxis",
    "MeshConfig",
    "MetricValue",
    "ParamRole",
    "PipelineParallelizableModule",
    "PipelineScheduleKind",
    "PipelineStepRunner",
    "PluginId",
    "PpStatus",
    "ProcessGroupManager",
    "RuntimeCore",
    "RuntimePhase",
    "RuntimePlugin",
    "RuntimeState",
    "SetupPhase",
    "StepContext",
    "StepRunner",
    "TpSpParallelizableModule",
    "acquire_buffer",
    "clear_buffer_pool",
    "global_buffer_pool",
    "release_buffer",
]


_EXPORTS = {
    "BufferHandle": ("runtime.buffer_allocator", "BufferHandle"),
    "BufferPolicy": ("runtime.buffer_allocator", "BufferPolicy"),
    "acquire_buffer": ("runtime.buffer_allocator", "acquire_buffer"),
    "clear_buffer_pool": ("runtime.buffer_allocator", "clear_buffer_pool"),
    "global_buffer_pool": ("runtime.buffer_allocator", "global_buffer_pool"),
    "release_buffer": ("runtime.buffer_allocator", "release_buffer"),
    "RuntimeCore": ("runtime.core", "RuntimeCore"),
    "MeshAxis": ("runtime.mesh", "MeshAxis"),
    "MeshConfig": ("runtime.mesh", "MeshConfig"),
    "ProcessGroupManager": ("runtime.mesh", "ProcessGroupManager"),
    "ContextParallelizableModule": ("runtime.plugin", "ContextParallelizableModule"),
    "ExpertParallelizableModule": ("runtime.plugin", "ExpertParallelizableModule"),
    "PipelineParallelizableModule": ("runtime.plugin", "PipelineParallelizableModule"),
    "PluginId": ("runtime.plugin", "PluginId"),
    "RuntimePlugin": ("runtime.plugin", "RuntimePlugin"),
    "TpSpParallelizableModule": ("runtime.plugin", "TpSpParallelizableModule"),
    "DefaultStepRunner": ("runtime.step_runners", "DefaultStepRunner"),
    "PipelineScheduleKind": ("runtime.step_runners", "PipelineScheduleKind"),
    "PipelineStepRunner": ("runtime.step_runners", "PipelineStepRunner"),
    "StepRunner": ("runtime.step_runners", "StepRunner"),
    "MetricValue": ("runtime.types", "MetricValue"),
    "ParamRole": ("runtime.types", "ParamRole"),
    "PpStatus": ("runtime.types", "PpStatus"),
    "RuntimePhase": ("runtime.types", "RuntimePhase"),
    "RuntimeState": ("runtime.types", "RuntimeState"),
    "SetupPhase": ("runtime.types", "SetupPhase"),
    "StepContext": ("runtime.types", "StepContext"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
