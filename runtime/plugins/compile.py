from __future__ import annotations

import torch
import torch.nn as nn

from parallel.protocols import CompilableModule
from runtime.plugin import PluginId, RuntimePlugin
from runtime.types import MetricValue, SetupPhase


class CompilePlugin(RuntimePlugin):
    """Compile a model-declared scope without wrapping the runtime model.

    ZeRO-3 and the parallel plugins attach execution hooks to modules and the
    checkpoint/state layer relies on stable parameter names.  Calling
    ``torch.compile(model)`` would wrap the root module and make those
    boundaries much harder to reason about.  Compiling only the selected
    module's ``forward`` keeps the module object, state dict, and
    ``nn.Module.__call__`` hook boundary intact.  In particular, ZeRO-3 must
    materialize and free full parameters outside the compiled graph because
    those hooks temporarily change parameter shapes.  The model, rather than
    this runtime plugin, declares which module paths are safe to compile.
    """

    def __init__(
        self,
        *,
        scope: str = "mlp",
        backend: str = "inductor",
        mode: str | None = "default",
    ) -> None:
        super().__init__(
            id=PluginId.COMPILE,
            name="compile",
            runs_after={
                PluginId.TP,
                PluginId.TP_SP,
                PluginId.CP,
                PluginId.HDP,
                PluginId.EP,
                PluginId.PP,
                PluginId.ZERO1,
                PluginId.ZERO2,
                PluginId.ZERO3,
                PluginId.FP16,
            },
        )
        if not scope:
            raise ValueError("compile scope must not be empty")
        self.scope = scope
        self.backend = backend
        self.mode = mode
        self.compiled_module_paths: list[str] = []

    def on_setup_phase(self, phase: SetupPhase, model: nn.Module) -> nn.Module:
        if phase != SetupPhase.FINALIZE:
            return model
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires torch.compile, which is unavailable in this PyTorch build")
        if not isinstance(model, CompilableModule):
            raise ValueError(
                f"model type={type(model).__name__} does not declare a compile_spec(); "
                "--compile requires the model architecture to opt in explicitly"
            )
        assert self.runtime is not None
        scopes = model.compile_spec()
        if not isinstance(scopes, dict) or not all(
            isinstance(scope_name, str)
            and isinstance(paths, list)
            and all(isinstance(path, str) for path in paths)
            for scope_name, paths in scopes.items()
        ):
            raise TypeError(
                "model compile_spec() must return dict[str, list[str]], "
                f"got {scopes!r}"
            )
        try:
            paths = scopes[self.scope]
        except KeyError as exc:
            available = ", ".join(sorted(scopes)) or "(none)"
            raise ValueError(
                f"compile scope={self.scope!r} is not declared by model type={type(model).__name__}; "
                f"available scopes: {available}"
            ) from exc
        targets: list[tuple[str, nn.Module]] = []
        for path in paths:
            if self.runtime.is_module_path_omitted(path):
                continue
            try:
                targets.append((path, model.get_submodule(path)))
            except AttributeError as exc:
                raise ValueError(
                    f"model compile_spec declared missing module path={path!r} for scope={self.scope!r}"
                ) from exc
        if not targets:
            raise ValueError(f"compile scope={self.scope!r} has no local modules after parallel transforms")
        for name, module in targets:
            # Keep Module.__call__ eager so parallel-runtime pre/post hooks are
            # not captured by Dynamo. ZeRO-3 hooks rebind parameters between
            # flat local shards and full-shaped tensors; Module.compile()
            # specializes those mutations to one shape. Replacing only the
            # bound forward preserves identity, hooks, and state keys.
            module.forward = torch.compile(
                module.forward,
                backend=self.backend,
                mode=self.mode,
            )
            self.compiled_module_paths.append(name)
        return model

    def collect_metrics(self) -> dict[str, MetricValue]:
        return {
            "compile/enabled": True,
            "compile/scope": self.scope,
            "compile/backend": self.backend,
            "compile/mode": self.mode,
            "compile/modules": len(self.compiled_module_paths),
        }
