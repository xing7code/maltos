"""Regression test: EP (no ZeRO) expert-grad sync must fire its EREP
all-reduce exactly once per real training step, on the fully-accumulated
gradient -- not once per grad-accumulation micro-step.

This targets a bug where ExpertParallelPlugin's bucket `pending` counter was
unconditionally re-armed to `len(bucket.params)` at every micro-step's
`backward_start` (instead of only at the step-boundary micro-step, mirroring
Zero1Plugin's `grad_accum_end`-gated arming). With grad_accum_steps > 1, that
fired a real EREP all-reduce on each micro-step's partial gradient instead of
firing once, at the end, on the fully-accumulated gradient.

A pure numerical comparison against a single-process reference is NOT a
reliable way to catch this: EP's all-to-all token routing already introduces
~1e-3-level floating point divergence from a non-parallel reference (routing
sums tokens through experts in a different order), which is the same order
of magnitude as the divergence this bug itself introduces -- so a value-diff
assertion can't reliably distinguish "correct but noisy" from "buggy" without
a carefully tuned mesh (correction != 1.0) and still risks flakiness. Instead
this test directly counts how many times the EREP-group `dist.all_reduce`
fires across two grad-accumulation micro-steps: it must be exactly 1 per
bucket, not 2.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from helpers import causal_lm_batch
from models import TinyMoETransformer
from parallel import ParallelPlan
from runtime import MeshAxis, MeshConfig, RuntimeCore
from runtime.plugins.ep import ExpertParallelPlugin

_MODEL_KWARGS = dict(
    dim=64,
    n_heads=4,
    n_kv_heads=4,
    hidden_size=128,
    eps=1e-5,
    n_layers=2,
    vocab_size=256,
    max_seq_len=64,
    num_experts=4,
)
_LR = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29612)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _tokens_for(seed: int, rank: int, microbatch: int, batch_size: int, seq_len: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed + rank * 100 + microbatch)
    return torch.randint(0, _MODEL_KWARGS["vocab_size"], (batch_size, seq_len), generator=generator)


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )
    mesh = MeshConfig(dp=args.world_size, tp=1, pp=1, cp=1, ep=args.ep_size)

    torch.manual_seed(args.seed)
    model = TinyMoETransformer(**_MODEL_KWARGS)
    ep_plugin = ExpertParallelPlugin()
    core = RuntimeCore(
        mesh=mesh,
        plan=ParallelPlan(),
        model=model,
        grad_accum_steps=2,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=[ep_plugin],
    )
    core.setup()
    core.model.train()

    erep_group = core.get_group(MeshAxis.EREP)
    if erep_group is None:
        raise AssertionError(
            "test setup is broken: no EREP group (need ep_size < dp_size so experts are replicated)"
        )

    call_count = 0
    orig_all_reduce = dist.all_reduce

    def counting_all_reduce(tensor: torch.Tensor, *fn_args: Any, group: Any = None, **fn_kwargs: Any) -> Any:
        nonlocal call_count
        if group is erep_group:
            call_count += 1
        return orig_all_reduce(tensor, *fn_args, group=group, **fn_kwargs)

    dist.all_reduce = counting_all_reduce
    try:
        for step in range(2):
            for microbatch in range(2):
                tokens = _tokens_for(
                    args.seed,
                    rank,
                    step * 2 + microbatch,
                    args.batch_size,
                    args.seq_len,
                )
                _, should_step = core.run_step(causal_lm_batch(tokens))
                expected_should_step = microbatch == 1
                if should_step != expected_should_step:
                    raise AssertionError(
                        f"step={step} microbatch={microbatch}: should_step={should_step}, "
                        f"expected {expected_should_step}"
                    )

            # The optimizer clears grads with set_to_none=True. The next
            # accumulation window must reattach every expert param to its flat
            # bucket buffer before backward writes the new gradients.
            if step == 1:
                for bucket in ep_plugin._grad_buckets:  # noqa: SLF001
                    offset = 0
                    for param in bucket.params:
                        expected = bucket.grad_buffer[
                            offset : offset + param.numel()
                        ].view_as(param)
                        if param.grad is None or param.grad.data_ptr() != expected.data_ptr():
                            raise AssertionError(
                                "expert param.grad was not reattached to its EP grad bucket "
                                "after optimizer.zero_grad(set_to_none=True)"
                            )
                        offset += param.numel()
            core.step_optimizer()
    finally:
        dist.all_reduce = orig_all_reduce

    num_buckets = len(ep_plugin._grad_buckets)  # noqa: SLF001

    payload = (rank, call_count, num_buckets)
    gathered: list = [None for _ in range(args.world_size)]
    dist.all_gather_object(gathered, payload)
    if rank == 0:
        ok = True
        for r, calls, buckets in gathered:
            expected_calls = 2 * buckets
            status = "OK" if calls == expected_calls else "FAIL"
            print(
                f"rank={r} {status} erep_all_reduce_calls={calls} "
                f"expected={expected_calls} (steps=2, num_buckets={buckets})"
            )
            ok = ok and calls == expected_calls
        if not ok:
            raise AssertionError(
                "EP fired the EREP all-reduce more than once per bucket across a 2-step grad-accum "
                "window -- it must fire exactly once (on the step-boundary micro-step, after both "
                "micro-batches' grads have accumulated), not once per micro-step. See module docstring."
            )
        print("PASS")

    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
