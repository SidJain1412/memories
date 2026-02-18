"""Embedder auto-reload decision policy with anti-loop guardrails."""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Callable, Deque, Dict


class EmbedderAutoReloadController:
    """Decides when embedder reload is allowed under pressure."""

    def __init__(
        self,
        rss_threshold_kb: int,
        required_high_streak: int = 3,
        min_interval_sec: float = 900.0,
        window_sec: float = 3600.0,
        max_per_window: int = 2,
        max_active_requests: int = 2,
        max_queue_depth: int = 0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.rss_threshold_kb = max(1, int(rss_threshold_kb))
        self.required_high_streak = max(1, int(required_high_streak))
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self.window_sec = max(1.0, float(window_sec))
        self.max_per_window = max(1, int(max_per_window))
        self.max_active_requests = max(0, int(max_active_requests))
        self.max_queue_depth = max(0, int(max_queue_depth))
        self._clock = clock or time.monotonic

        self._lock = threading.Lock()
        self._high_streak = 0
        self._last_reload_monotonic: float | None = None
        self._reload_times: Deque[float] = deque()

    def evaluate(self, rss_kb: int, active_requests: int, queue_depth: int) -> Dict[str, object]:
        now = self._clock()
        rss = max(0, int(rss_kb))
        active = max(0, int(active_requests))
        queue = max(0, int(queue_depth))

        with self._lock:
            while self._reload_times and (now - self._reload_times[0]) > self.window_sec:
                self._reload_times.popleft()

            if rss < self.rss_threshold_kb:
                self._high_streak = 0
                return {
                    "trigger": False,
                    "reason": "below_threshold",
                    "rss_kb": rss,
                    "high_streak": self._high_streak,
                }

            if active > self.max_active_requests:
                self._high_streak = 0
                return {
                    "trigger": False,
                    "reason": "busy_requests",
                    "rss_kb": rss,
                    "active_requests": active,
                    "max_active_requests": self.max_active_requests,
                }

            if queue > self.max_queue_depth:
                self._high_streak = 0
                return {
                    "trigger": False,
                    "reason": "busy_queue",
                    "rss_kb": rss,
                    "queue_depth": queue,
                    "max_queue_depth": self.max_queue_depth,
                }

            self._high_streak += 1
            if self._high_streak < self.required_high_streak:
                return {
                    "trigger": False,
                    "reason": "high_streak",
                    "rss_kb": rss,
                    "high_streak": self._high_streak,
                    "required_high_streak": self.required_high_streak,
                }

            elapsed = 0.0
            if self._last_reload_monotonic is not None:
                elapsed = now - self._last_reload_monotonic
            if self._last_reload_monotonic is not None and elapsed < self.min_interval_sec:
                return {
                    "trigger": False,
                    "reason": "cooldown",
                    "rss_kb": rss,
                    "seconds_until_next": round(self.min_interval_sec - elapsed, 3),
                }

            if len(self._reload_times) >= self.max_per_window:
                return {
                    "trigger": False,
                    "reason": "window_cap",
                    "rss_kb": rss,
                    "window_sec": self.window_sec,
                    "max_per_window": self.max_per_window,
                }

            self._last_reload_monotonic = now
            self._reload_times.append(now)
            self._high_streak = 0
            return {
                "trigger": True,
                "reason": "triggered",
                "rss_kb": rss,
                "reloads_in_window": len(self._reload_times),
            }
