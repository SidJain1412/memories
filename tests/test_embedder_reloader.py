"""Tests for embedder auto-reload controller guardrails."""

from __future__ import annotations

from embedder_reloader import EmbedderAutoReloadController


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, sec: float) -> None:
        self.now += sec


def test_requires_high_streak_before_trigger() -> None:
    clock = FakeClock()
    controller = EmbedderAutoReloadController(
        rss_threshold_kb=1000,
        required_high_streak=3,
        min_interval_sec=60.0,
        window_sec=3600.0,
        max_per_window=2,
        max_active_requests=2,
        max_queue_depth=0,
        clock=clock,
    )

    assert controller.evaluate(rss_kb=900, active_requests=0, queue_depth=0)["trigger"] is False
    assert controller.evaluate(rss_kb=1300, active_requests=0, queue_depth=0)["trigger"] is False
    assert controller.evaluate(rss_kb=1300, active_requests=0, queue_depth=0)["trigger"] is False

    result = controller.evaluate(rss_kb=1300, active_requests=0, queue_depth=0)
    assert result["trigger"] is True
    assert result["reason"] == "triggered"


def test_enforces_cooldown_and_window_cap() -> None:
    clock = FakeClock()
    controller = EmbedderAutoReloadController(
        rss_threshold_kb=1000,
        required_high_streak=1,
        min_interval_sec=10.0,
        window_sec=60.0,
        max_per_window=2,
        max_active_requests=2,
        max_queue_depth=0,
        clock=clock,
    )

    assert controller.evaluate(rss_kb=1300, active_requests=0, queue_depth=0)["trigger"] is True

    clock.advance(2.0)
    cooldown = controller.evaluate(rss_kb=1300, active_requests=0, queue_depth=0)
    assert cooldown["trigger"] is False
    assert cooldown["reason"] == "cooldown"

    clock.advance(10.0)
    assert controller.evaluate(rss_kb=1300, active_requests=0, queue_depth=0)["trigger"] is True

    clock.advance(10.0)
    capped = controller.evaluate(rss_kb=1300, active_requests=0, queue_depth=0)
    assert capped["trigger"] is False
    assert capped["reason"] == "window_cap"

    clock.advance(61.0)
    assert controller.evaluate(rss_kb=1300, active_requests=0, queue_depth=0)["trigger"] is True


def test_skips_when_service_is_busy() -> None:
    clock = FakeClock()
    controller = EmbedderAutoReloadController(
        rss_threshold_kb=1000,
        required_high_streak=1,
        min_interval_sec=0.0,
        window_sec=60.0,
        max_per_window=5,
        max_active_requests=1,
        max_queue_depth=0,
        clock=clock,
    )

    busy = controller.evaluate(rss_kb=1300, active_requests=3, queue_depth=0)
    assert busy["trigger"] is False
    assert busy["reason"] == "busy_requests"

    queued = controller.evaluate(rss_kb=1300, active_requests=0, queue_depth=2)
    assert queued["trigger"] is False
    assert queued["reason"] == "busy_queue"

    ready = controller.evaluate(rss_kb=1300, active_requests=0, queue_depth=0)
    assert ready["trigger"] is True
    assert ready["reason"] == "triggered"
