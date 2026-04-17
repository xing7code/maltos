from __future__ import annotations

from dataclasses import dataclass, field

from train_system.parallel.plan import ParallelPlan


@dataclass
class RuntimeContext:
    plan: ParallelPlan
    plugins: list = field(default_factory=list)

    def validate(self) -> None:
        self.plan.validate()

    def register_plugin(self, plugin) -> None:
        self.plugins.append(plugin)
