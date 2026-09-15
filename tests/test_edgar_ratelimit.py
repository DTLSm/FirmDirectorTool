"""The rate limiter is the one piece of this client that must not be wrong.

Every test here drives a fake clock rather than sleeping, so the suite stays
fast and the assertions are exact instead of approximate.
"""

import pytest

from firmdirectortool.edgar import TokenBucket


class FakeClock:
    """A clock that only advances when something sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_bucket(clock: FakeClock, rate: float = 10.0, capacity: float | None = None) -> TokenBucket:
    return TokenBucket(rate, capacity, monotonic=clock.monotonic, sleep=clock.sleep)


def test_burst_is_free(clock: FakeClock) -> None:
    bucket = make_bucket(clock, rate=10.0, capacity=10.0)
    for _ in range(10):
        assert bucket.acquire() == 0.0
    assert clock.slept == []


def test_eleventh_request_waits_one_tenth_of_a_second(clock: FakeClock) -> None:
    bucket = make_bucket(clock, rate=10.0, capacity=10.0)
    for _ in range(10):
        bucket.acquire()
    waited = bucket.acquire()
    assert waited == pytest.approx(0.1)


def test_sustained_rate_matches_the_configured_limit(clock: FakeClock) -> None:
    """100 requests at 10/s with a burst of 10 must take (100 - 10) / 10 seconds."""
    bucket = make_bucket(clock, rate=10.0, capacity=10.0)
    start = clock.now
    for _ in range(100):
        bucket.acquire()
    assert clock.now - start == pytest.approx(9.0)


def test_tokens_refill_over_idle_time_but_do_not_exceed_capacity(clock: FakeClock) -> None:
    bucket = make_bucket(clock, rate=10.0, capacity=10.0)
    for _ in range(10):
        bucket.acquire()
    clock.now += 60.0  # idle for a minute
    for _ in range(10):
        assert bucket.acquire() == 0.0
    assert bucket.acquire() == pytest.approx(0.1)


def test_asking_for_more_than_capacity_raises_rather_than_hanging(clock: FakeClock) -> None:
    bucket = make_bucket(clock, rate=10.0, capacity=10.0)
    with pytest.raises(ValueError):
        bucket.acquire(11)


def test_rejects_nonsense_configuration(clock: FakeClock) -> None:
    with pytest.raises(ValueError):
        make_bucket(clock, rate=0)
    with pytest.raises(ValueError):
        make_bucket(clock, rate=10.0, capacity=0.5)
