"""Regression test: EP groups nested inside a single TP group (ep < tp).

This targets a specific mesh shape that none of the existing EP+TP+SP
equivalence cases exercise: `ep_size < tp_size` with `reuse_tp_for_ep=True`
(the default). All existing matrix entries use `ep_size == tp_size` (so EREP
is trivial/singleton) or `ep_size == world_size` with `tp_size == 1` (no TP
group to conflict with). Here we use tp=4, ep=2, dp=1, cp=1, so:

    flat = tp_idx  (cp=dp=1)
    EP group   = flat // ep  -> {0,1} holds experts {0,1}; {2,3} holds experts {2,3}
    EREP group = flat % ep   -> {0,2} and {1,3}

EREP is a strict, non-trivial subset of the TP group {0,1,2,3}. SequenceParallelPlugin
used to register a post-grad-reduction callback (role_filter=EXPERT) that all-reduced
over `self.sp_group` (the full TP group) instead of MeshAxis.EREP. That mixed in
gradients from ranks holding entirely different experts (e.g. rank 0, which owns
expert 0, would sum in rank 1's and rank 3's local_experts gradients even though
rank 1/3 hold experts 1 and 3) -- a correctness bug, not just wasted bandwidth. The
callback was redundant besides: ExpertParallelPlugin (without ZeRO) and each ZeRO
plugin already reduce expert grads correctly over MeshAxis.EREP themselves. Fixed by
deleting the SP callback entirely (runtime/plugins/sp.py) and giving
ExpertParallelPlugin's own non-ZeRO EREP all-reduce the same seq-slice/DP-replica
correction factor ZeRO already used (`expert_erep_correction` in
runtime/plugins/zero_common.py).

We do not have TP-sharded expert weights in the tiny MoE model, so a plain TP=4
model would only have the SP-wrapped attention sharded; the MoE weights are full
replicas of local_experts on each EP group. The relevant comparison is against a
single-process baseline model with no parallelism at all: baseline expert `e`'s
gradient must match the runtime's local_experts gradient on the rank(s) that own
expert `e`, without contamination from ranks that hold different experts.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distributed_test_utils import max_diff as _max_diff
from helpers import causal_lm_batch
from models import TinyMoETransformer, TinyMoETransformerTpSp
from parallel import ParallelPlan
from runtime import MeshConfig, RuntimeCore
from runtime.types import RuntimePhase
from runtime.plugins.ep import ExpertParallelPlugin, _ExpertParallelMoE
from runtime.plugins.sp import SequenceParallelPlugin
from runtime.plugins.tp import TensorParallelPlugin

_MODEL_KWARGS = dict(
    dim=32,
    n_heads=4,
    n_kv_heads=4,
    hidden_size=64,
    eps=1e-5,
    n_layers=1,
    vocab_size=64,
    max_seq_len=16,
    num_experts=4,
)

_LR = 1e-2
# Expert grads that are contaminated by an unrelated EREP-subgroup all-reduce
# are wrong by O(1) (they sum in an entirely different expert's gradient), so a
# tight tolerance is enough to distinguish "correct" from "bug reproduced".
_GRAD_ATOL = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29591)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _run_worker(rank: int, args: argparse.Namespace) -> None:
    dist.init_process_group(
        backend=args.backend,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=args.world_size,
    )

    mesh = MeshConfig(dp=1, tp=args.tp_size, pp=1, cp=1, ep=args.ep_size)

    torch.manual_seed(args.seed)
    baseline_model = TinyMoETransformer(**_MODEL_KWARGS)
    tokens = torch.randint(0, _MODEL_KWARGS["vocab_size"], (args.batch_size, args.seq_len), generator=torch.Generator().manual_seed(args.seed))

    baseline_model.train()
    baseline_model.zero_grad(set_to_none=True)
    baseline_loss = baseline_model(causal_lm_batch(tokens))
    baseline_loss.backward()
    baseline_expert_grads: dict[str, torch.Tensor] = {}
    for layer_idx, layer in enumerate(baseline_model.layers):
        for expert_idx, expert in enumerate(layer.moe.experts):
            for pname, p in expert.named_parameters():
                key_name = f"layers.{layer_idx}.moe.experts.{expert_idx}.{pname}"
                baseline_expert_grads[key_name] = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)

    sharded_model = TinyMoETransformerTpSp(**_MODEL_KWARGS)
    sharded_model.load_state_dict(baseline_model.state_dict())

    plugins = [
        TensorParallelPlugin(),
        SequenceParallelPlugin(),
        ExpertParallelPlugin(),
    ]
    core = RuntimeCore(
        mesh=mesh,
        plan=ParallelPlan(),
        model=sharded_model,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=_LR),
        plugins=plugins,
    )
    core.setup()
    core.model.train()

    loss, should_step = core.run_step(causal_lm_batch(tokens))
    if not should_step:
        raise AssertionError("expected should_step=True")
    core._run_phase(RuntimePhase.PRE_STEP)

    runtime_expert_grads: dict[str, torch.Tensor] = {}
    for module_name, module in core.model.named_modules():
        if not isinstance(module, _ExpertParallelMoE):
            continue
        for local_idx, global_idx in enumerate(module.local_expert_ids):
            expert = module.local_experts[local_idx]
            for pname, p in expert.named_parameters():
                key_name = f"{module_name}.experts.{global_idx}.{pname}"
                runtime_expert_grads[key_name] = (
                    p.grad.detach().clone().cpu() if p.grad is not None else torch.zeros_like(p).cpu()
                )

    # Compare only the experts this rank actually owns -- that's exactly the
    # set of grads that must equal the single-process baseline with no
    # contamination from ranks holding other experts.
    local_baseline = {name: baseline_expert_grads[name] for name in runtime_expert_grads}
    grad_name, grad_diff = _max_diff(local_baseline, runtime_expert_grads)

    ok = grad_diff <= _GRAD_ATOL
    payload = (rank, ok, grad_name, grad_diff)
    gathered: list = [None for _ in range(args.world_size)]
    dist.all_gather_object(gathered, payload)

    if rank == 0:
        all_ok = True
        for r, ok_r, name_r, diff_r in gathered:
            status = "OK" if ok_r else "FAIL"
            print(f"rank={r} {status} worst_param={name_r} diff={diff_r:.4e} (atol={_GRAD_ATOL:.1e})")
            all_ok = all_ok and ok_r
        if not all_ok:
            raise AssertionError(
                "EP expert-gradient equivalence failed under ep<tp mesh (tp=4, ep=2). "
                "See module docstring for the EREP-subgroup contamination bug this "
                "regression test guards against. See details above."
            )
        print("PASS")

    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    assert args.world_size == args.tp_size
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mp.spawn(_run_worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
