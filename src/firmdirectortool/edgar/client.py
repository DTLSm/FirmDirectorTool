"""The EDGAR HTTP client.

Thin on purpose. It owns four things and nothing else:

* the ``User-Agent`` the SEC requires,
* the rate limit (see :mod:`.ratelimit`),
* retry-with-backoff on transient failures,
* the raw response cache (see :mod:`.cache`).

Everything above it — which index to walk, which form types to keep, how to
read an ``ownershipDocument`` — lives in other modules and talks to EDGAR only
through this class. That boundary is what makes it possible to run the whole
ingestion pipeline against recorded fixtures with no network at all: swap the
transport, keep every other line of code.

Not using an EDGAR library for this layer is a deliberate choice. The parsing
is arguably better done by someone else (see :mod:`.parse`), but the HTTP
behaviour *is* the operational story of this project, and a library that
silently retries or silently does not is a bad thing to have under a scheduled
job.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx

from .cache import RawCache
from .errors import (
    BlockedError,
    EdgarConnectionError,
    EdgarHTTPError,
    NotFoundError,
    RateLimitError,
)
from .ratelimit import TokenBucket

#: The SEC's published fair-access ceiling, in requests per second.
SEC_RATE_LIMIT = 10.0

#: Statuses worth trying again. 403 is excluded on purpose: see :class:`BlockedError`.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class ClientConfig:
    """Everything tunable about the client.

    Defaults are the values to run in production; tests override them to make
    backoff instantaneous.
    """

    user_agent: str
    rate_per_second: float = SEC_RATE_LIMIT
    burst: float = SEC_RATE_LIMIT
    max_retries: int = 5
    backoff_base: float = 0.5
    backoff_cap: float = 30.0
    max_retry_after: float = 60.0
    timeout: float = 30.0
    cache_root: Path | None = Path("data/raw")

    def __post_init__(self) -> None:
        if "@" not in self.user_agent:
            raise ValueError(
                "EDGAR requires a User-Agent containing a contact email address, "
                f'e.g. "Jane Doe jane@example.org"; got {self.user_agent!r}'
            )
        if self.rate_per_second > SEC_RATE_LIMIT:
            raise ValueError(
                f"rate_per_second={self.rate_per_second} exceeds the SEC fair-access "
                f"limit of {SEC_RATE_LIMIT}/s"
            )

    @classmethod
    def from_env(cls) -> ClientConfig:
        """Build from the environment.

        ``EDGAR_USER_AGENT`` is required and has no default. Hard-coding a
        contact address into a public repository would put it in every image,
        every log line and every fork, and the address is a real one that the
        SEC uses to get in touch when a client misbehaves.

        ``EDGAR_CACHE_ROOT`` and ``EDGAR_RATE_PER_SECOND`` are optional.
        """
        user_agent = os.environ.get("EDGAR_USER_AGENT", "").strip()
        if not user_agent:
            raise RuntimeError(
                "EDGAR_USER_AGENT is not set. The SEC requires a descriptive "
                'User-Agent with a contact email, e.g. "Jane Doe jane@example.org". '
                "See .env.example."
            )
        cache_root = os.environ.get("EDGAR_CACHE_ROOT", "data/raw").strip()
        rate = os.environ.get("EDGAR_RATE_PER_SECOND", "").strip()
        return cls(
            user_agent=user_agent,
            rate_per_second=float(rate) if rate else SEC_RATE_LIMIT,
            cache_root=Path(cache_root) if cache_root else None,
        )


def retry_after_seconds(response: httpx.Response) -> float | None:
    """Interpret a ``Retry-After`` header, or return ``None`` if there is none.

    The header has two legal forms (RFC 9110): a number of seconds, or an
    HTTP-date. Both appear in the wild. Handle both; on anything unparseable
    return ``None`` and let the caller fall back to its own backoff rather than
    raising — a malformed header is not a reason to abandon the request.

    For the date form, the delay is the date minus *now*; clamp negatives to
    zero, and be careful to compare timezone-aware datetimes.
    """
    value = response.headers.get("Retry-After", "").strip()
    if not value:
        return None

    # delay-seconds is 1*DIGIT. Checking the characters rather than calling
    # float() keeps out "-5", "1e9", "nan" and "inf", all of which float()
    # would happily accept and none of which are a sane amount of time to sleep.
    if value.isascii() and value.isdigit():
        return float(int(value))

    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    # A "-0000" offset parses to a naive datetime. RFC 9110 dates are always
    # GMT, so read it as UTC rather than letting the subtraction below raise.
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def backoff_delay(
    attempt: int,
    *,
    base: float,
    cap: float,
    rng: random.Random | None = None,
) -> float:
    """Seconds to wait before retry number ``attempt`` (0-based).

    Exponential — ``base * 2 ** attempt`` — clamped at ``cap``, with *full
    jitter*: return a uniform draw from ``[0, that]`` rather than the value
    itself.

    Jitter matters more than the exponent here. Without it, a fleet of clients
    that all hit a 503 at the same moment retry at the same moment, and keep
    doing so in lockstep; the retries themselves become the outage. Full jitter
    spreads them out. This is one client today, but the ingestion job will run
    as parallel Kubernetes pods by Slice 5.

    ``rng`` is injectable so tests can pin the draw.
    """
    ceiling = min(cap, base * 2**attempt)
    uniform = rng.uniform if rng is not None else random.uniform
    return uniform(0.0, ceiling)


class EdgarClient:
    """A rate-limited, retrying, caching HTTP client for EDGAR.

    Parameters
    ----------
    config
        Defaults to :meth:`ClientConfig.from_env`.
    transport
        Injected for tests. Pass an :class:`httpx.MockTransport` to run the
        whole stack without a network.
    sleep
        Injected for tests, so backoff does not actually take 30 seconds.

    Use as a context manager; the underlying connection pool needs closing.
    """

    def __init__(
        self,
        config: ClientConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or ClientConfig.from_env()
        self.cache = RawCache(self.config.cache_root) if self.config.cache_root else None
        self.bucket = TokenBucket(
            self.config.rate_per_second,
            self.config.burst,
            sleep=sleep,
        )
        self._sleep = sleep
        self._http = httpx.Client(
            headers={
                "User-Agent": self.config.user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=self.config.timeout,
            transport=transport,
            follow_redirects=True,
        )

    # ------------------------------------------------------------ lifecycle

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -------------------------------------------------------------- fetching

    def get_bytes(self, url: str, *, use_cache: bool = True) -> bytes:
        """Fetch ``url`` and return the body.

        The order of operations is the whole design, so spell it out:

        1. If ``use_cache`` and the cache has it, return it. **Before** touching
           the rate limiter — a cache hit must not spend request budget, or a
           re-parse of a year of filings takes as long as the original fetch.
        2. ``self.bucket.acquire()``.
        3. Send. On a retryable status (:data:`RETRYABLE_STATUSES`), sleep for
           :func:`retry_after_seconds` if the server said so, otherwise
           :func:`backoff_delay`, and go back to step 2 — the retry is a new
           request and must pay for a new token. If ``Retry-After`` asks for
           longer than ``config.max_retry_after``, stop retrying instead of
           sleeping: retrying early earns another 429, and a scheduled run is
           better failing now and being rerun than stalling for an hour.
        4. Map what is left onto the exception hierarchy: 403 to
           :class:`~.errors.BlockedError` immediately with no retry, 404 to
           :class:`~.errors.NotFoundError`, an exhausted budget on 429 to
           :class:`~.errors.RateLimitError`, anything else non-2xx to
           :class:`~.errors.EdgarHTTPError`.
        5. Write to the cache, then return.

        Connection-level failures (:class:`httpx.TransportError`) are retryable
        on the same schedule — a reset connection is not different from a 503
        as far as this loop is concerned. Once the budget is spent they surface
        as :class:`~.errors.EdgarConnectionError`, so a caller catching
        :class:`~.errors.EdgarError` sees every way a fetch can fail.

        Note that ``use_cache=False`` still *writes* to the cache. The flag
        means "do not trust what is there", which is what a daily index for the
        current day needs; it does not mean "do not record what came back".
        """
        if use_cache and self.cache is not None:
            cached = self.cache.get(url)
            if cached is not None:
                return cached

        max_retries = self.config.max_retries
        attempt = 0
        gave_up_because: str | None = None
        while True:
            self.bucket.acquire()
            delay: float | None
            try:
                response = self._http.get(url)
            except httpx.TransportError as exc:
                if attempt >= max_retries:
                    raise EdgarConnectionError(
                        url, f"no response from {url} after {attempt + 1} attempts: {exc!r}"
                    ) from exc
                delay = None
            else:
                if response.status_code not in RETRYABLE_STATUSES or attempt >= max_retries:
                    break
                delay = retry_after_seconds(response)
                if delay is not None and delay > self.config.max_retry_after:
                    gave_up_because = (
                        f"HTTP {response.status_code} from {url} with Retry-After of "
                        f"{delay:.0f}s, over max_retry_after={self.config.max_retry_after:g}s"
                    )
                    break
            if delay is None:
                delay = backoff_delay(
                    attempt, base=self.config.backoff_base, cap=self.config.backoff_cap
                )
            self._sleep(delay)
            attempt += 1

        # The loop only exits on a response it will not retry: a success, a
        # terminal status, or a retryable one with the budget spent or a
        # Retry-After too long to wait out.
        status = response.status_code
        if status == 403:
            raise BlockedError(url)
        if status == 404:
            raise NotFoundError(status, url)
        if status == 429:
            raise RateLimitError(status, url, gave_up_because)
        if not response.is_success:
            raise EdgarHTTPError(status, url, gave_up_because)

        body = response.content
        if self.cache is not None:
            self.cache.put(url, body)
        return body

    def get_text(self, url: str, *, use_cache: bool = True, encoding: str = "utf-8") -> str:
        """:meth:`get_bytes`, decoded. EDGAR serves latin-1 in places; errors are replaced."""
        return self.get_bytes(url, use_cache=use_cache).decode(encoding, errors="replace")

    def get_json(self, url: str, *, use_cache: bool = True) -> Any:
        """:meth:`get_bytes`, parsed as JSON."""
        return json.loads(self.get_bytes(url, use_cache=use_cache))


__all__ = [
    "RETRYABLE_STATUSES",
    "SEC_RATE_LIMIT",
    "BlockedError",
    "ClientConfig",
    "EdgarClient",
    "EdgarConnectionError",
    "EdgarHTTPError",
    "NotFoundError",
    "RateLimitError",
    "backoff_delay",
    "retry_after_seconds",
]
