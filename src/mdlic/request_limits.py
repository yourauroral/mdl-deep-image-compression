"""Small in-process request limiter for the single-worker demo service."""

from __future__ import annotations

import math
import threading
import time
from collections import deque


class SlidingWindowRateLimiter:
    """Bounded per-key sliding-window limiter.

    This is intentionally process-local. Public multi-worker deployments must
    also enforce a shared limit at the reverse proxy.
    """

    def __init__(self, limit: int, window_seconds: float, max_keys: int = 4096):
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if max_keys <= 0:
            raise ValueError("max_keys must be positive")
        self.limit = limit
        self.window_seconds = float(window_seconds)
        self.max_keys = max_keys
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def admit(self, key: str, *, now: float | None = None) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds) for one request."""
        if self.limit == 0:
            return True, 0
        timestamp = time.monotonic() if now is None else float(now)
        cutoff = timestamp - self.window_seconds

        with self._lock:
            for existing_key in list(self._events):
                events = self._events[existing_key]
                while events and events[0] <= cutoff:
                    events.popleft()
                if not events:
                    del self._events[existing_key]

            if key not in self._events:
                if len(self._events) >= self.max_keys:
                    return False, max(1, math.ceil(self.window_seconds))
                self._events[key] = deque()

            events = self._events[key]
            if len(events) >= self.limit:
                retry_after = events[0] + self.window_seconds - timestamp
                return False, max(1, math.ceil(retry_after))
            events.append(timestamp)
            return True, 0
