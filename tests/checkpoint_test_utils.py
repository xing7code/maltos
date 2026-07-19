from __future__ import annotations


def test_runtime_spec(label: str = "test") -> dict[str, object]:
    return {
        "format": "test_runtime_spec",
        "version": 1,
        "label": label,
    }
