from __future__ import annotations

from runtime.plugins.metrics import MetricPlugin


def test_metric_plugin_emits_step_measurements_once() -> None:
    plugin = MetricPlugin()
    plugin._metrics["perf/step_sec"] = 1.25

    assert plugin.collect_metrics() == {"perf/step_sec": 1.25}
    assert plugin.collect_metrics() == {}


if __name__ == "__main__":
    test_metric_plugin_emits_step_measurements_once()
    print("PASS")
