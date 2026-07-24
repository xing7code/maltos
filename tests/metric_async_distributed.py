from __future__ import annotations

import os
import sys
import tempfile

import torch.distributed as dist
import torch.multiprocessing as mp

from utils.metrics import MetricAggregator


def _worker(rank: int, init_method: str) -> None:
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        aggregator = MetricAggregator()
        aggregator.update(
            {
                "step": 7,
                "loss": 1.0 + 2.0 * rank,
                "train/tokens": 10.0 * (rank + 1),
                "memory/allocated_gb": 2.0 + 3.0 * rank,
                "fp16/overflow": rank == 1,
            }
        )
        pending = aggregator.flush_async(step_delta=1)

        # The work has been issued, but it need not be observed until a later
        # training boundary.  This is the path Trainer uses for log lag.
        metrics = pending.wait()
        assert metrics["step"] == 7
        assert metrics["loss"] == 2.0
        assert metrics["train/tokens"] == 30.0
        assert metrics["memory/allocated_gb"] == 5.0
        assert metrics["fp16/overflow"] is True
    finally:
        dist.destroy_process_group()


def main() -> None:
    # macOS Gloo does not always select the loopback interface automatically.
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    with tempfile.NamedTemporaryFile(delete=False) as store:
        init_method = f"file://{store.name}"
    try:
        mp.spawn(_worker, args=(init_method,), nprocs=2, join=True)
    finally:
        if os.path.exists(store.name):
            os.unlink(store.name)
    print("async distributed metrics: PASS")


if __name__ == "__main__":
    main()
