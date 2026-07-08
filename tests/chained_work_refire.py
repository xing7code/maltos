"""Regression test for persistent ChainedWork replay across training steps."""

from __future__ import annotations

from runtime.plugins.zero_common import ChainedWork


class _FakeWork:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def block_current_stream(self) -> None:
        self.events.append(f"block:{self.name}")

    def wait(self) -> None:
        self.events.append(f"wait:{self.name}")


def main() -> None:
    events: list[str] = []
    fire_counts = {"root": 0, "leaf": 0}

    def make_functor(name: str):
        def functor() -> _FakeWork:
            fire_counts[name] += 1
            invocation = fire_counts[name]
            events.append(f"fire:{name}:{invocation}")
            return _FakeWork(f"{name}:{invocation}", events)

        return functor

    root = ChainedWork(None, make_functor("root"))
    leaf = ChainedWork(root, make_functor("leaf"))

    for _ in range(2):
        leaf.fire()
        leaf.wait()

    expected_events = [
        "fire:root:1",
        "block:root:1",
        "fire:leaf:1",
        "wait:leaf:1",
        "fire:root:2",
        "block:root:2",
        "fire:leaf:2",
        "wait:leaf:2",
    ]
    if events != expected_events:
        raise AssertionError(
            "persistent ChainedWork did not replay the full chain on every fire:\n"
            f"expected={expected_events}\nactual={events}"
        )
    print("PASS")


if __name__ == "__main__":
    main()
