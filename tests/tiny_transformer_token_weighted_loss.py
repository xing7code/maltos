"""Equivalence test: token-weighted loss normalization under grad accumulation.

Without --token-weighted-loss each micro-batch contributes 1/A of the step
gradient regardless of how many supervised tokens it holds, so the tokens of a
sparsely-labelled micro-batch are over-weighted by (mean n / its own n).  With
the flag on, the step gradient must equal the gradient of

    sum_i CE_sum_i / sum_i n_i

which this test computes directly in one process, with no accumulation at all.

The two micro-batches are built with deliberately different label counts, so
the "equal weight per micro-batch" and "equal weight per token" answers differ.

Usage:
  PYTHONPATH=.:tests .venv/bin/python tests/tiny_transformer_token_weighted_loss.py
"""

from __future__ import annotations

import torch

from models import TinyTransformer
from parallel import ParallelPlan
from runtime import MeshConfig, RuntimeCore
from utils.constants import IGNORE_INDEX, INPUT_IDS_KEY, LABELS_KEY

_KW = dict(dim=64, n_heads=4, n_kv_heads=4, hidden_size=128, eps=1e-5,
           n_layers=2, vocab_size=256, max_seq_len=64)
_SEQ = 32
_ATOL = 2e-6


def _batches(seed: int):
    """Two micro-batches whose supervised-token counts differ by ~4x."""
    torch.manual_seed(seed)
    out = []
    for keep in (28, 7):
        ids = torch.randint(0, _KW["vocab_size"], (1, _SEQ))
        labels = ids.clone()
        labels[:, : _SEQ - keep] = IGNORE_INDEX  # mask the "prompt"
        out.append({INPUT_IDS_KEY: ids, LABELS_KEY: labels})
    return out


def _reference_grads(batches) -> dict[str, torch.Tensor]:
    """Gradient of (sum of CE over every supervised token) / (total such tokens)."""
    torch.manual_seed(0)
    model = TinyTransformer(**_KW)
    model.train()
    model.zero_grad(set_to_none=True)

    total_tokens = sum(int(b[LABELS_KEY].ne(IGNORE_INDEX).sum()) for b in batches)
    for b in batches:
        n = int(b[LABELS_KEY].ne(IGNORE_INDEX).sum())
        mean_ce = model(b)                      # mean over this batch's own targets
        (mean_ce * n / total_tokens).backward()  # -> summed CE / global token count
    return {k: v.grad.detach().clone() for k, v in model.named_parameters()}


def _runtime_grads(batches, *, token_weighted: bool) -> dict[str, torch.Tensor]:
    """Effective gradient recovered from the parameter delta of one SGD step.

    Going through step_optimizer() rather than reading .grad keeps the whole
    path under test, including where the normalizer is invoked.
    """
    torch.manual_seed(0)
    model = TinyTransformer(**_KW)
    core = RuntimeCore(
        mesh=MeshConfig(dp=1, tp=1, pp=1, cp=1, ep=1),
        plan=ParallelPlan(),
        model=model,
        grad_accum_steps=len(batches),
        token_weighted_loss=token_weighted,
        nominal_tokens_per_microbatch=_SEQ,
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=1.0),
    )
    core.setup()
    core.model.train()
    before = {k: v.detach().clone() for k, v in core.model.named_parameters()}
    should_step = False
    for b in batches:
        _, should_step = core.run_step(b)
    assert should_step, "expected the last micro-batch to land on the step boundary"
    core.step_optimizer()
    grads = {k: before[k] - v.detach() for k, v in core.model.named_parameters()}
    core.close()
    return grads


def _worst(a, b) -> tuple[str, float]:
    name, worst = "", 0.0
    for k, v in a.items():
        d = (v - b[k]).abs().max().item()
        if d > worst:
            name, worst = k, d
    return name, worst


def main() -> None:
    batches = _batches(42)
    counts = [int(b[LABELS_KEY].ne(IGNORE_INDEX).sum()) for b in batches]
    print(f"supervised tokens per micro-batch : {counts}  (ratio {max(counts)/min(counts):.1f}x)")

    ref = _reference_grads(batches)
    off = _runtime_grads(batches, token_weighted=False)
    on = _runtime_grads(batches, token_weighted=True)

    n_off, d_off = _worst(off, ref)
    n_on, d_on = _worst(on, ref)
    print(f"divide-by-accum-count  vs reference : {d_off:.3e}  ({n_off})")
    print(f"token-weighted         vs reference : {d_on:.3e}  ({n_on})")
    print()

    if d_on > _ATOL:
        raise AssertionError(
            f"token-weighted gradients do not match the token-weighted reference: "
            f"param={n_on}, diff={d_on:.3e}, atol={_ATOL:.1e}"
        )
    if d_off <= _ATOL:
        raise AssertionError(
            "the two normalizations agree, so this test proves nothing -- "
            "the micro-batches need different supervised-token counts"
        )
    print(f"PASS: token-weighted matches the reference; dividing by the accumulation "
          f"count is off by {d_off:.3e}")


if __name__ == "__main__":
    main()
