"""Index URL construction, record parsing, and the day-by-day walk."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest

from firmdirectortool.edgar import (
    ClientConfig,
    EdgarClient,
    IndexEntry,
    daily_index_url,
    parse_form_idx,
    quarter_of,
    quarterly_index_url,
    walk_daily,
)

UA = "Test Runner test@example.org"
CHIME = "0001193125-26-389357"


# ------------------------------------------------------------------- urls


def test_quarter_of() -> None:
    assert [quarter_of(date(2026, m, 1)) for m in (1, 3, 4, 6, 7, 9, 10, 12)] == [
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
    ]


def test_daily_index_url() -> None:
    assert daily_index_url(date(2026, 9, 11)) == (
        "https://www.sec.gov/Archives/edgar/daily-index/2026/QTR3/form.20260911.idx"
    )


def test_quarterly_index_url() -> None:
    assert quarterly_index_url(2026, 2) == (
        "https://www.sec.gov/Archives/edgar/full-index/2026/QTR2/form.idx"
    )
    with pytest.raises(ValueError):
        quarterly_index_url(2026, 5)


# ---------------------------------------------------------------- parsing


@pytest.fixture
def daily(fixtures: Path) -> list[IndexEntry]:
    return list(parse_form_idx((fixtures / "form.20260911.idx").read_text()))


@pytest.fixture
def quarterly(fixtures: Path) -> list[IndexEntry]:
    return list(parse_form_idx((fixtures / "form.2026-QTR2.idx").read_text()))


def test_daily_preamble_and_headers_are_skipped(daily: list[IndexEntry]) -> None:
    """The daily header wraps over two lines; neither is a record."""
    assert len(daily) == 25
    assert all(e.company_name for e in daily)


def test_daily_record_is_read_exactly(daily: list[IndexEntry]) -> None:
    ambev = next(e for e in daily if e.company_name == "AMBEV S.A.")
    assert ambev.form_type == "3"
    assert ambev.cik == 1565025
    assert ambev.date_filed == date(2026, 9, 11)
    assert ambev.filename == "edgar/data/1565025/0001193125-26-389339.txt"


def test_form_type_may_contain_a_space(daily: list[IndexEntry]) -> None:
    """`1-A POS` is one form type, not a form type and a company name."""
    entry = next(e for e in daily if e.cik == 2025795)
    assert entry.form_type == "1-A POS"
    assert entry.company_name == "McQueen Labs Series, LLC"


def test_quarterly_uses_dashed_dates_and_a_single_header_line(
    quarterly: list[IndexEntry],
) -> None:
    assert len(quarterly) == 30
    first = quarterly[0]
    assert first.form_type == "1"
    assert first.company_name == ("Reserved for Notice Registration of Security Futures Product")
    assert first.cik == 2132551
    assert first.date_filed == date(2026, 4, 29)


def test_accession_is_derived_from_the_filename(daily: list[IndexEntry]) -> None:
    entry = next(e for e in daily if e.company_name == "Chime Financial, Inc.")
    assert entry.accession == CHIME
    assert entry.accession_nodash == "000119312526389357"
    assert entry.url == ("https://www.sec.gov/Archives/edgar/data/1795586/0001193125-26-389357.txt")
    assert entry.folder_url == (
        "https://www.sec.gov/Archives/edgar/data/1795586/000119312526389357"
    )


def test_one_filing_appears_once_per_filer(daily: list[IndexEntry]) -> None:
    """Eleven lines, eleven CIKs, eleven URLs — and one document."""
    lines = [e for e in daily if e.accession == CHIME]
    assert len(lines) == 11
    assert len({e.cik for e in lines}) == 11
    assert len({e.url for e in lines}) == 11
    assert len({e.accession for e in lines}) == 1


def test_a_corrupt_record_is_not_swallowed(fixtures: Path) -> None:
    from firmdirectortool.edgar import ParseError

    text = (fixtures / "form.20260911.idx").read_text()
    broken = text.replace("1565025     20260911", "NOT-A-CIK   20260911")
    with pytest.raises(ParseError):
        list(parse_form_idx(broken))


# ------------------------------------------------------------------- walk


class DailyIndexServer:
    """Serves the fixture for 11 Sep 2026 and 404s every other day."""

    def __init__(self, body: str) -> None:
        self.body = body
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url == daily_index_url(date(2026, 9, 11)):
            return httpx.Response(200, text=self.body)
        return httpx.Response(404)


def make_client(tmp_path: Path, server: DailyIndexServer) -> EdgarClient:
    config = ClientConfig(
        user_agent=UA, backoff_base=0.01, backoff_cap=0.02, cache_root=tmp_path / "raw"
    )
    return EdgarClient(config, transport=httpx.MockTransport(server), sleep=lambda _: None)


def test_walk_skips_days_with_no_index(fixtures: Path, tmp_path: Path) -> None:
    """Weekends and holidays 404. That is not an error."""
    server = DailyIndexServer((fixtures / "form.20260911.idx").read_text())
    with make_client(tmp_path, server) as client:
        entries = list(walk_daily(client, date(2026, 9, 9), date(2026, 9, 13)))
    assert len(server.requests) == 5
    assert len(entries) == 19  # forms 3 and 4 and 4/A only


def test_walk_filters_on_exact_form_type(fixtures: Path, tmp_path: Path) -> None:
    """`1` must not match `1-A/A` or `10-K`."""
    server = DailyIndexServer((fixtures / "form.20260911.idx").read_text())
    with make_client(tmp_path, server) as client:
        ones = list(walk_daily(client, date(2026, 9, 11), date(2026, 9, 11), form_types={"1"}))
        fours = list(walk_daily(client, date(2026, 9, 11), date(2026, 9, 11), form_types={"4"}))
    assert ones == []
    assert len(fours) == 15


def test_walk_does_not_trust_a_cached_copy_of_todays_index(fixtures: Path, tmp_path: Path) -> None:
    """Today's index is still being appended to; yesterday's is immutable."""
    day = date(2026, 9, 11)
    server = DailyIndexServer((fixtures / "form.20260911.idx").read_text())
    with make_client(tmp_path, server) as client:
        list(walk_daily(client, day, day, today=day))
        list(walk_daily(client, day, day, today=day))
        assert len(server.requests) == 2

        list(walk_daily(client, day, day, today=date(2026, 9, 15)))
        assert len(server.requests) == 2  # served from cache


# -------------------------------------------------------------- quarterly


class QuarterlyIndexServer:
    """Serves the 2026 Q2 fixture for every quarter asked, and records which."""

    def __init__(self, body: str) -> None:
        self.body = body
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        return httpx.Response(200, text=self.body)


def test_walk_quarterly_fetches_each_overlapping_quarter_once(
    fixtures: Path, tmp_path: Path
) -> None:
    from firmdirectortool.edgar import walk_quarterly

    server = QuarterlyIndexServer((fixtures / "form.2026-QTR2.idx").read_text())
    config = ClientConfig(user_agent=UA, cache_root=tmp_path / "raw")
    with EdgarClient(config, transport=httpx.MockTransport(server), sleep=lambda _: None) as c:
        list(walk_quarterly(c, date(2025, 11, 15), date(2026, 5, 1), form_types={"1-A"}))
    assert server.requests == [
        quarterly_index_url(2025, 4),
        quarterly_index_url(2026, 1),
        quarterly_index_url(2026, 2),
    ]


def test_walk_quarterly_honours_the_date_window_exactly(fixtures: Path, tmp_path: Path) -> None:
    """The quarter has 1-A filings on many days; only those inside [start, end] come back."""
    from firmdirectortool.edgar import walk_quarterly

    text = (fixtures / "form.2026-QTR2.idx").read_text()
    server = QuarterlyIndexServer(text)
    config = ClientConfig(user_agent=UA, cache_root=tmp_path / "raw")
    start, end = date(2026, 4, 23), date(2026, 5, 13)
    with EdgarClient(config, transport=httpx.MockTransport(server), sleep=lambda _: None) as c:
        got = list(walk_quarterly(c, start, end, form_types={"1-A"}))
    expected = [
        e for e in parse_form_idx(text) if e.form_type == "1-A" and start <= e.date_filed <= end
    ]
    assert got == expected
    assert got  # the window is not empty
    assert len(got) < sum(e.form_type == "1-A" for e in parse_form_idx(text))
