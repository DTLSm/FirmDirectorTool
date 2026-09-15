"""Client behaviour, driven entirely through httpx.MockTransport.

No test here touches the network. That is the payoff of letting the transport
be injected: the retry logic, the cache and the rate limiter can all be
exercised exactly, including the failure paths that are hard to provoke against
a real server.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest

from firmdirectortool.edgar import (
    BlockedError,
    ClientConfig,
    EdgarClient,
    EdgarHTTPError,
    NotFoundError,
    RateLimitError,
    backoff_delay,
    retry_after_seconds,
)

URL = "https://www.sec.gov/Archives/edgar/data/320193/0000320193-26-000008.txt"
UA = "Test Runner test@example.org"

Step = tuple[int, bytes, dict[str, str]] | BaseException


def ok(body: bytes = b"a document") -> Step:
    return (200, body, {})


def fail(code: int, headers: dict[str, str] | None = None) -> Step:
    return (code, b"", headers or {})


class Script:
    """Replays scripted responses; the last one repeats forever."""

    def __init__(self, *steps: Step) -> None:
        self.steps = list(steps)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, BaseException):
            raise step
        code, body, headers = step
        return httpx.Response(code, content=body, headers=headers)


def make_client(
    tmp_path: Path, script: Script, *, max_retries: int = 3
) -> tuple[EdgarClient, list[float]]:
    slept: list[float] = []
    config = ClientConfig(
        user_agent=UA,
        backoff_base=0.01,
        backoff_cap=0.05,
        max_retries=max_retries,
        cache_root=tmp_path / "raw",
    )
    client = EdgarClient(config, transport=httpx.MockTransport(script), sleep=slept.append)
    return client, slept


# --------------------------------------------------------------------- config


def test_config_requires_a_contact_email() -> None:
    with pytest.raises(ValueError, match="email"):
        ClientConfig(user_agent="firmdirectortool/0.1")


def test_config_refuses_to_exceed_the_sec_fair_access_limit() -> None:
    with pytest.raises(ValueError, match="fair-access"):
        ClientConfig(user_agent=UA, rate_per_second=25)


def test_from_env_requires_the_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="EDGAR_USER_AGENT"):
        ClientConfig.from_env()


def test_from_env_reads_the_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGAR_USER_AGENT", UA)
    assert ClientConfig.from_env().user_agent == UA


# -------------------------------------------------------------------- fetching


def test_sends_the_user_agent(tmp_path: Path) -> None:
    script = Script(ok())
    client, _ = make_client(tmp_path, script)
    with client:
        client.get_bytes(URL)
    assert script.requests[0].headers["user-agent"] == UA


def test_returns_the_body_and_caches_it(tmp_path: Path) -> None:
    script = Script(ok(b"<ownershipDocument/>"))
    client, _ = make_client(tmp_path, script)
    with client:
        assert client.get_bytes(URL) == b"<ownershipDocument/>"
        assert client.cache is not None
        assert client.cache.get(URL) == b"<ownershipDocument/>"


def test_cache_hit_spends_neither_a_request_nor_a_token(tmp_path: Path) -> None:
    script = Script(ok())
    client, slept = make_client(tmp_path, script)
    with client:
        client.get_bytes(URL)
        before = client.bucket.tokens
        for _ in range(50):
            client.get_bytes(URL)
    assert len(script.requests) == 1
    assert client.bucket.tokens == before
    assert slept == []


def test_use_cache_false_refetches_but_still_records(tmp_path: Path) -> None:
    """What today's daily index needs: distrust the copy, keep the new one."""
    script = Script(ok(b"morning"), ok(b"afternoon"))
    client, _ = make_client(tmp_path, script)
    with client:
        assert client.get_bytes(URL) == b"morning"
        assert client.get_bytes(URL, use_cache=False) == b"afternoon"
        assert client.get_bytes(URL) == b"afternoon"
    assert len(script.requests) == 2


# --------------------------------------------------------------------- retries


def test_retries_after_429(tmp_path: Path) -> None:
    script = Script(fail(429), fail(429), ok(b"finally"))
    client, slept = make_client(tmp_path, script)
    with client:
        assert client.get_bytes(URL) == b"finally"
    assert len(script.requests) == 3
    assert len(slept) == 2


def test_honours_retry_after_over_its_own_backoff(tmp_path: Path) -> None:
    script = Script(fail(429, {"Retry-After": "2"}), ok())
    client, slept = make_client(tmp_path, script)
    with client:
        client.get_bytes(URL)
    assert slept == [pytest.approx(2.0)]


def test_gives_up_on_a_persistent_429(tmp_path: Path) -> None:
    script = Script(fail(429))
    client, _ = make_client(tmp_path, script, max_retries=3)
    with client, pytest.raises(RateLimitError):
        client.get_bytes(URL)
    assert len(script.requests) == 4  # the first try plus three retries


def test_retries_server_errors(tmp_path: Path) -> None:
    script = Script(fail(503), fail(500), ok(b"up again"))
    client, _ = make_client(tmp_path, script)
    with client:
        assert client.get_bytes(URL) == b"up again"


def test_retries_a_dropped_connection(tmp_path: Path) -> None:
    script = Script(httpx.ConnectError("reset by peer"), ok(b"second attempt"))
    client, _ = make_client(tmp_path, script)
    with client:
        assert client.get_bytes(URL) == b"second attempt"


def test_403_is_terminal(tmp_path: Path) -> None:
    """A block is made worse by retrying. Fail fast and say why."""
    script = Script(fail(403))
    client, _ = make_client(tmp_path, script)
    with client, pytest.raises(BlockedError, match="User-Agent"):
        client.get_bytes(URL)
    assert len(script.requests) == 1


def test_404_is_terminal_and_distinguishable(tmp_path: Path) -> None:
    """walk_daily relies on catching exactly this for weekends and holidays."""
    script = Script(fail(404))
    client, _ = make_client(tmp_path, script)
    with client, pytest.raises(NotFoundError):
        client.get_bytes(URL)
    assert len(script.requests) == 1


def test_other_client_errors_raise(tmp_path: Path) -> None:
    script = Script(fail(400))
    client, _ = make_client(tmp_path, script)
    with client, pytest.raises(EdgarHTTPError):
        client.get_bytes(URL)


def test_get_json_decodes(tmp_path: Path) -> None:
    script = Script(ok(json.dumps({"cik": "320193"}).encode()))
    client, _ = make_client(tmp_path, script)
    with client:
        assert client.get_json(URL) == {"cik": "320193"}


# --------------------------------------------------------------------- backoff


def test_backoff_grows_exponentially_and_is_capped() -> None:
    rng = random.Random(1)
    for attempt in range(8):
        delay = backoff_delay(attempt, base=0.5, cap=8.0, rng=rng)
        assert 0.0 <= delay <= min(8.0, 0.5 * 2**attempt)


def test_backoff_is_jittered() -> None:
    """Without jitter, a fleet of retrying clients stays in lockstep."""
    rng = random.Random(7)
    draws = {backoff_delay(5, base=0.5, cap=30.0, rng=rng) for _ in range(30)}
    assert len(draws) > 1


def test_backoff_uses_the_injected_rng() -> None:
    first = backoff_delay(3, base=0.5, cap=30.0, rng=random.Random(42))
    second = backoff_delay(3, base=0.5, cap=30.0, rng=random.Random(42))
    assert first == second


# ---------------------------------------------------------------- retry-after


def test_retry_after_in_seconds() -> None:
    assert retry_after_seconds(httpx.Response(429, headers={"Retry-After": "120"})) == 120.0


def test_retry_after_as_an_http_date() -> None:
    when = datetime.now(UTC) + timedelta(seconds=30)
    response = httpx.Response(503, headers={"Retry-After": format_datetime(when, usegmt=True)})
    delay = retry_after_seconds(response)
    assert delay is not None
    assert delay == pytest.approx(30.0, abs=2.0)


def test_retry_after_in_the_past_clamps_to_zero() -> None:
    when = datetime.now(UTC) - timedelta(hours=1)
    response = httpx.Response(503, headers={"Retry-After": format_datetime(when, usegmt=True)})
    assert retry_after_seconds(response) == 0.0


def test_retry_after_absent_or_unparseable_is_none() -> None:
    assert retry_after_seconds(httpx.Response(429)) is None
    assert retry_after_seconds(httpx.Response(429, headers={"Retry-After": "soonish"})) is None
