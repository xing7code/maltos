from __future__ import annotations

import copy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from models import LlamaConfig, LlamaForCausalLM
from models.llama import LlamaRMSNorm
from runtime import MeshConfig, RuntimeCore
from runtime.plugins.compile import CompilePlugin
from runtime.plugins.zero3 import Zero3Plugin


def _model() -> LlamaForCausalLM:
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=16,
        )
    )


def _single_process_equivalence() -> None:
    torch.manual_seed(123)
    eager_model = _model()
    compiled_model = copy.deepcopy(eager_model)
    eager = RuntimeCore(mesh=MeshConfig(), model=eager_model, optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.0))
    compiled = RuntimeCore(
        mesh=MeshConfig(),
        model=compiled_model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.0),
        plugins=[CompilePlugin(backend="eager")],
    )
    eager.setup()
    compiled.setup()
    assert len(next(plugin for plugin in compiled.plugins if isinstance(plugin, CompilePlugin)).compiled_module_paths) == 2
    assert all(not name.startswith("_orig_mod.") for name in compiled.model.state_dict())

    batch = {
        "input_ids": torch.randint(0, 64, (2, 8)),
        "labels": torch.randint(0, 64, (2, 8)),
    }
    eager_loss, eager_should_step = eager.run_step(batch)
    compiled_loss, compiled_should_step = compiled.run_step(batch)
    assert eager_should_step and compiled_should_step
    torch.testing.assert_close(eager_loss, compiled_loss, rtol=1e-5, atol=1e-6)
    for (name, eager_param), (_, compiled_param) in zip(eager.model.named_parameters(), compiled.model.named_parameters()):
        assert eager_param.grad is not None and compiled_param.grad is not None, name
        torch.testing.assert_close(eager_param.grad, compiled_param.grad, rtol=1e-5, atol=1e-6)
    eager.close()
    compiled.close()


def _hooks_stay_outside_compiled_forward() -> None:
    class HookBoundaryModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 2))
            self.forward_was_compiled = False

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.forward_was_compiled = torch._dynamo.is_compiling()
            return value @ self.weight

    module = HookBoundaryModule()
    hook_states: list[bool] = []
    module.register_forward_pre_hook(lambda *_: hook_states.append(torch._dynamo.is_compiling()))
    module.register_forward_hook(lambda *_: hook_states.append(torch._dynamo.is_compiling()))
    module.forward = torch.compile(module.forward, backend="eager")

    result = module(torch.ones(1, 2))
    torch.testing.assert_close(result, torch.full((1, 2), 2.0))
    assert module.forward_was_compiled
    assert hook_states == [False, False]


def _zero3_worker(rank: int, port: int) -> None:
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        torch.manual_seed(456)
        runtime = RuntimeCore(
            mesh=MeshConfig(dp=2),
            model=_model(),
            optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.0),
            plugins=[
                Zero3Plugin({nn.Linear, nn.Embedding, LlamaRMSNorm}),
                CompilePlugin(backend="eager"),
            ],
        )
        runtime.setup()
        zero3 = next(plugin for plugin in runtime.plugins if isinstance(plugin, Zero3Plugin))
        compile_plugin = next(plugin for plugin in runtime.plugins if isinstance(plugin, CompilePlugin))
        for path in compile_plugin.compiled_module_paths:
            module = runtime.model.get_submodule(path)
            target_param_ids = {id(param) for param in module.parameters()}
            target_buckets = [
                bucket
                for bucket in zero3.buckets
                if target_param_ids.intersection(id(param) for param in bucket.params)
            ]
            assert target_buckets
            assert all(bucket.module is module for bucket in target_buckets)
            assert {
                id(param)
                for bucket in target_buckets
                for param in bucket.params
            } == target_param_ids
        loss, should_step = runtime.run_step(
            {
                "input_ids": torch.randint(0, 64, (2, 8)),
                "labels": torch.randint(0, 64, (2, 8)),
            }
        )
        assert should_step
        assert torch.isfinite(loss)
        runtime.step_optimizer()
        assert len(compile_plugin.compiled_module_paths) == 2
        runtime.close()
    finally:
        dist.destroy_process_group()


def main() -> None:
    _hooks_stay_outside_compiled_forward()
    _single_process_equivalence()
    mp.spawn(_zero3_worker, args=(29663,), nprocs=2, join=True)
    print("compile plugin ok")


if __name__ == "__main__":
    main()
