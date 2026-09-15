"""Token-bucket rate limiting.

The SEC publishes a fair-access limit of 10 requests per second, counted
across *all* of its hosts — ``www.sec.gov`` and ``data.sec.gov`` share one
budget. Exceeding it earns a 403 block that outlives the process.

**Why the limiter belongs to the client rather than the caller.** A budget is a
property of the remote service, not of any one piece of calling code. If every
caller throttled itself, the ingestion job, the backfill script and an
interactive notebook would each stay under 10 rps and collectively be at 30.
Putting one bucket behind one client object makes the invariant hold no matter
how many callers there are, and makes it impossible to forget. The same
argument is why the bucket is thread-safe: a thread pool fanning out over
accession numbers is the obvious next optimisation, and it must not be able to
breach the limit.

A token bucket rather than a fixed sleep because it permits a short burst —
useful when a caller makes two requests and then does a second of parsing —
while holding the long-run average at the configured rate.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

#: Tolerance for floating-point dust when comparing tokens.
#:
#: Refilling multiplies an elapsed time by a rate, and neither is exact in
#: binary: after a wait computed to land on exactly one token, the bucket may
#: hold 0.9999999999999998 instead. Without this slack the loop below would
#: compute a new delay of about 1e-16, sleep for a duration too small to change
#: the clock at all, find the same shortfall, and spin forever.
TOKEN_EPSILON = 1e-9


class TokenBucket:
    """A thread-safe token bucket.

    Tokens refill continuously at ``rate`` per second up to ``capacity``.
    :meth:`acquire` blocks until the requested tokens are available.

    Parameters
    ----------
    rate
        Tokens added per second. For EDGAR this is the request rate.
    capacity
        Maximum tokens held, i.e. the largest instantaneous burst. Defaults to
        ``rate`` (one second of budget).
    monotonic
        Clock source. Injected so tests can drive time deterministically rather
        than sleeping. Must be monotonic; wall-clock time can go backwards.
    sleep
        Sleep function. Injected for the same reason.
    """

    def __init__(
        self,
        rate: float,
        capacity: float | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = float(rate)
        self.capacity = float(capacity) if capacity is not None else float(rate)
        if self.capacity < 1:
            raise ValueError(f"capacity must be at least 1 token, got {self.capacity}")
        self._monotonic = monotonic
        self._sleep = sleep
        self._tokens = self.capacity
        self._updated = monotonic()
        self._lock = threading.Lock()

    @property
    def tokens(self) -> float:
        """Tokens currently in the bucket, without refilling. For tests."""
        return self._tokens

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available, then consume them.

        Returns the number of seconds spent waiting, so callers can log or
        assert on throttling without timing the call themselves.
        """
        if tokens > self.capacity:
            # Waiting would never succeed, however long we slept. Say so now
            # rather than hanging forever.
            raise ValueError(
                f"cannot acquire {tokens} tokens from a bucket that holds at most {self.capacity}"
            )

        waited = 0.0
        # The lock is held across the whole check-sleep-recheck cycle, not just
        # the arithmetic. Releasing it while sleeping would let two threads each
        # conclude there was room for the same token. Serialising the waits is
        # the point: the bucket *is* the queue.
        with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens - TOKEN_EPSILON:
                    self._tokens = max(0.0, self._tokens - tokens)
                    return waited
                # Exactly how long the missing tokens take to appear.
                delay = (tokens - self._tokens) / self.rate
                self._sleep(delay)
                waited += delay
                # Loop rather than assuming the sleep was exact: time.sleep is
                # allowed to overshoot, and a fake clock in a test may not
                # advance the way we asked.

    def _refill(self) -> None:
        """Add the tokens that accrued since the last call. Caller holds the lock.

        There is no background thread topping the bucket up; it is refilled
        lazily on demand, which is both cheaper and easier to reason about.
        """
        now = self._monotonic()
        elapsed = max(0.0, now - self._updated)
        self._updated = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
