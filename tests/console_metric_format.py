from __future__ import annotations

from utils.metrics import _format_metrics


def test_training_console_line_is_compact_and_readable() -> None:
    line = _format_metrics(
        {
            "step": 7,
            "loss": 0.071682,
            "lr": 4.5e-6,
            "perf/step_sec": 34.0181,
            "train/tokens_per_sec": 7706.01,
            "perf/tflops_per_gpu": 85.9974,
            "zero3/grad_norm": 0.417559,
            "memory/allocated_gb": 53.0127,
            "memory/max_allocated_gb": 64.5874,
        }
    )
    assert line == (
        "[step 00007] loss 0.071682 | lr 4.50e-06 | step 34.02s | "
        "throughput 7,706 tok/s | compute 86.0 TF/GPU | grad-norm 0.4176 | memory 53.0/64.6 GiB"
    )


def test_console_format_falls_back_for_non_training_metrics() -> None:
    assert _format_metrics({"foo": 1, "loss/aux": 0.5}) == "foo=1 loss/aux=0.5"


if __name__ == "__main__":
    test_training_console_line_is_compact_and_readable()
    test_console_format_falls_back_for_non_training_metrics()
    print("PASS")
