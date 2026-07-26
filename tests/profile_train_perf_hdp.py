"""CPU coverage for HDP synthetic prefetch and paper-summary reproducibility."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from data import PrefetchDataLoader
from data.bytescale_hdp import ByteScaleHdpDataLoader
from parallel.hdp_helper import ByteScaleHdpBalancedConfig
from tools.profile_train_perf import (
    SyntheticBatchConfig,
    SyntheticBatchStream,
    _PrefetchMetrics,
    _TimedHdpSyntheticLoader,
    _configure_benchmark_args,
    _load_cases,
    _load_training_args,
    _trace_summary,
    _write_summary,
)


TRACE = ((4096,), (2048, 1024, 512))


def _synthetic() -> SyntheticBatchConfig:
    return SyntheticBatchConfig(
        layout="packed", segment_count_min=2, segment_count_max=4,
        supervised_fraction_min=0.55, supervised_fraction_max=0.70,
        max_padding_fraction=0.0, delivery="host", document_length_trace=TRACE,
    )


def test_prefetch_wraps_hdp_materialization_and_restores_consumed_cursor() -> None:
    stream = SyntheticBatchStream(
        batch_size=2, seq_len=4096, vocab_size=128, config=_synthetic(),
        seed=17, runtime_device=__import__("torch").device("cpu"),
    )
    metrics = _PrefetchMetrics()
    loader = _TimedHdpSyntheticLoader(
        ByteScaleHdpDataLoader(
            stream, hdp_rank=0, hdp_world_size=8,
            config=ByteScaleHdpBalancedConfig(partition_tokens=512),
        ),
        metrics,
    )
    prefetch = PrefetchDataLoader(loader)
    try:
        first = prefetch.next_batch()
        assert len(first["bytescale_hdp_waves"]) >= 1
        state = prefetch.state_dict()
        assert state == {"index": 1}, state
        second = prefetch.next_batch()  # consumes the prefetch-worker batch
        assert len(second["bytescale_hdp_waves"]) >= 1
    finally:
        prefetch.close()
    snapshot = metrics.snapshot()
    assert snapshot["scheduler_materialization_batches"] >= 2
    assert snapshot["prefetch_thread_batches"] >= 1
    assert prefetch._future is None
    assert prefetch._executor._shutdown


def test_trace_summary_and_json_summary_are_reproducible_without_cuda() -> None:
    cases = _load_cases(Path("configs/profile_train_perf_cases.yaml"))
    case = cases["olmo2_13b_hdp_flash_gqa_packed_8xa100_paper"]
    args = _load_training_args(
        case, ["--hdp-cost-alpha", "0.25", "--hdp-cost-beta", "1.5", "--hdp-cost-gamma", "3.0"]
    )
    trace = _trace_summary(_synthetic(), args, synthetic_seed=1234)
    assert trace is not None
    assert trace["active_tokens_per_microbatch"] == 7680
    assert trace["nominal_tokens_per_microbatch"] == 8192
    assert trace["hdp_degree_histogram_per_microbatch"] == {"1": 1, "2": 1, "4": 1, "8": 1}
    with tempfile.TemporaryDirectory() as directory:
        _write_summary(
            output_dir=Path(directory), case=case, train_args=args, synthetic=_synthetic(),
            data_source="synthetic", data_format=None, data_paths=None, warmup=20,
            step_seconds=[2.0, 2.0], peak_memory_bytes=0.0, world_size=8,
            final_loss=1.0, profiled=False, prefetch_metrics={"prefetch_thread_batches": 1},
            synthetic_seed=1234,
        )
        payload = json.loads((Path(directory) / "summary.json").read_text())
    assert payload["throughput_nominal_tokens_per_second"] == 4096.0
    assert payload["throughput_actual_active_tokens_per_second"] == 3840.0
    assert payload["hdp"]["cost_alpha"] == 0.25
    assert payload["synthetic"]["document_length_trace_sha256"] == trace["trace_sha256"]
    assert payload["synthetic"]["seed"] == 1234
    assert payload["trace_workload"]["seed"] == 1234
    assert "tflops_per_gpu" not in payload
    assert payload["hdp"]["loader_prefetch"]["scope"].startswith("all_produced_batches")


def test_paper_case_rejects_uncalibrated_cost_proxy() -> None:
    cases = _load_cases(Path("configs/profile_train_perf_cases.yaml"))
    case = cases["olmo2_13b_hdp_flash_gqa_packed_8xa100_paper"]
    args = _load_training_args(case, [])
    tool_args = argparse.Namespace(
        packed_input=None, synthetic_layout=None, packed_sequences=None,
        segment_count_min=None, segment_count_max=None,
        supervised_fraction_min=None, supervised_fraction_max=None,
        max_padding_fraction=None, synthetic_delivery=None, warmup=None, steps=None,
        use_recipe_optimizer=False, benchmark_lr=0.0, profile=False,
        profile_all_ranks=False, profile_record_shapes=False, profile_memory=False,
        data_source=None,
    )
    try:
        _configure_benchmark_args(args, tool_args, case)
    except ValueError as exc:
        assert "requires calibrated" in str(exc)
    else:
        raise AssertionError("paper case accepted the uncalibrated 1/0/0 proxy")


def main() -> None:
    test_prefetch_wraps_hdp_materialization_and_restores_consumed_cursor()
    test_trace_summary_and_json_summary_are_reproducible_without_cuda()
    test_paper_case_rejects_uncalibrated_cost_proxy()
    print("profile train perf HDP PASS")


if __name__ == "__main__":
    main()
