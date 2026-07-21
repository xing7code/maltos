"""Regression tests for the TinyMoE top-1 router balance loss.

Usage:
  PYTHONPATH=. .venv/bin/python tests/tiny_moe_aux_loss.py
"""

from __future__ import annotations

import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from models import TinyMoETransformer, Top1MoE
from runtime import MeshConfig, RuntimeCore
from runtime.layers.moe import ExpertParallelMoE
from runtime.plugins.pp import PipelineParallelPlugin
from runtime.types import LossOutput
from parallel import ParallelPlan
from parallel.plan import PipelineScheduleConfig


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_formula_and_runtime_metrics() -> None:
    torch.manual_seed(7)
    router_test = Top1MoE(dim=4, hidden_size=8, num_experts=4)
    torch.nn.init.zeros_(router_test.router.weight)
    _output, exact_aux = router_test(torch.randn(2, 3, 4), return_aux_loss=True)
    # Uniform probabilities plus deterministic top-1 tie-breaking to expert 0:
    # E * f_0 * p_0 = 4 * 1 * 1/4 = 1.
    torch.testing.assert_close(exact_aux, torch.tensor(1.0))
    exact_aux.backward()
    assert router_test.router.weight.grad is not None

    model = TinyMoETransformer(
        dim=16,
        n_heads=4,
        n_kv_heads=4,
        hidden_size=32,
        eps=1e-5,
        n_layers=2,
        vocab_size=32,
        max_seq_len=8,
        num_experts=4,
        moe_aux_loss_coef=0.01,
    )
    batch = (
        torch.randint(0, 32, (2, 8)),
        torch.randint(0, 32, (2, 8)),
    )

    output = model(batch)
    assert isinstance(output, LossOutput)
    assert output.loss.requires_grad
    assert output.metrics["moe/load_balance_loss"] > 0
    assert output.metrics["loss/total"] > output.metrics["loss/ce"]

    core = RuntimeCore(
        mesh=MeshConfig(dp=1, tp=1, pp=1, cp=1, ep=1),
        model=model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=1e-2),
    )
    core.setup()
    _loss, should_step = core.run_step(batch)
    assert should_step
    metrics = core.collect_metrics()
    assert metrics["moe/load_balance_loss"] > 0
    assert metrics["loss/total"] == metrics["loss"]
    print("formula/runtime metrics: PASS")


def _run_ep_worker(rank: int, port: int) -> None:
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        torch.manual_seed(11)
        moe = Top1MoE(dim=8, hidden_size=16, num_experts=4)
        ep_moe = ExpertParallelMoE.from_moe(moe, dist.group.WORLD)
        x = torch.randn(2, 3, 8)
        expected_output, expected_aux = moe(x, return_aux_loss=True)
        actual_output, actual_aux = ep_moe(x, return_aux_loss=True)
        torch.testing.assert_close(actual_output, expected_output)
        torch.testing.assert_close(actual_aux, expected_aux)
        actual_aux.backward()
        assert ep_moe.router.weight.grad is not None
        print(f"rank {rank}: EP aux loss PASS")
    finally:
        dist.destroy_process_group()


def test_ep_transformed_moe() -> None:
    mp.spawn(_run_ep_worker, args=(_free_port(),), nprocs=2, join=True)


def _run_pp_worker(rank: int, port: int, schedule: str) -> None:
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        torch.manual_seed(23)
        kwargs = dict(
            dim=16,
            n_heads=4,
            n_kv_heads=4,
            hidden_size=32,
            eps=1e-5,
            n_layers=4,
            vocab_size=32,
            max_seq_len=8,
            num_experts=4,
            moe_aux_loss_coef=0.05,
        )
        reference = TinyMoETransformer(**kwargs)
        runtime_model = TinyMoETransformer(**kwargs)
        runtime_model.load_state_dict(reference.state_dict())
        batch = (
            torch.randint(0, 32, (4, 8)),
            torch.randint(0, 32, (4, 8)),
        )
        reference_loss = torch.zeros(())
        for microbatch_idx in range(2):
            microbatch = tuple(value.narrow(0, microbatch_idx * 2, 2).contiguous() for value in batch)
            reference_output = reference(microbatch)
            assert isinstance(reference_output, LossOutput)
            (reference_output.loss / 2).backward()
            reference_loss = reference_loss + reference_output.loss.detach() / 2
        expected_grads = {
            name: param.grad.detach().clone()
            for name, param in reference.named_parameters()
            if param.grad is not None and ".moe.router." in name
        }

        core = RuntimeCore(
            mesh=MeshConfig(dp=1, tp=1, pp=2, cp=1, ep=1),
            plan=ParallelPlan(pp_schedule=PipelineScheduleConfig(microbatches=2)),
            model=runtime_model,
            optimizer_factory=lambda params: torch.optim.SGD(params, lr=1e-2),
            plugins=[PipelineParallelPlugin(schedule=schedule)],
        )
        core.setup()
        loss, should_step = core.run_step(batch)
        assert should_step
        torch.testing.assert_close(loss, reference_loss, atol=1e-6, rtol=1e-6)
        for name, param in core.model.named_parameters():
            if ".moe.router." in name:
                assert param.grad is not None, name
                torch.testing.assert_close(param.grad, expected_grads[name], atol=1e-6, rtol=1e-6)
        metrics = core.collect_metrics()
        torch.testing.assert_close(torch.tensor(metrics["loss/total"]), reference_loss)
        print(f"rank {rank}: PP {schedule} aux loss PASS")
    finally:
        dist.destroy_process_group()


def test_pp_aux_loss() -> None:
    for schedule in ("afab", "1f1b"):
        mp.spawn(_run_pp_worker, args=(_free_port(), schedule), nprocs=2, join=True)


def main() -> None:
    test_formula_and_runtime_metrics()
    test_ep_transformed_moe()
    test_pp_aux_loss()
    print("tiny MoE aux-loss regression PASS")


if __name__ == "__main__":
    main()
