"""The ingestion job, run end to end against the fixtures with no network.

The daily index for 11 Sep 2026 has 19 Section 16 lines naming 5 accessions,
and the fixture directory holds the submission file for each of those 5. So
one day's walk is a complete, offline, realistic run.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest

from firmdirectortool.edgar import (
    ClientConfig,
    EdgarClient,
    EdgarHTTPError,
    IndexEntry,
    daily_index_url,
    parse_form_idx,
)
from firmdirectortool.ingest import (
    MemoryStore,
    Outcome,
    group_by_accession,
    ingest_window,
    process_accession,
)

UA = "Test Runner test@example.org"
DAY = date(2026, 9, 11)
CHIME = "0001193125-26-389357"
NWPX = "0001437749-26-030170"

#: Accession -> fixture stem, for the five filings the daily index names.
FIXTURE_FOR = {
    "0001193125-26-389339": "form3_ambev",
    "0000863110-26-000091": "form4a_artesian",
    NWPX: "form4_nwpx",
    "0000100885-26-000268": "form4_union_pacific",
    CHIME: "form4_chime_multi_owner",
}


class FixtureEdgar:
    """Serves the daily index for 11 Sep and the five submission files it names."""

    def __init__(self, fixtures: Path) -> None:
        self.fixtures = fixtures
        self.requests: list[str] = []
        self.broken: set[str] = set()  # accessions to serve with the XML gutted
        self.failing: set[str] = set()  # accessions to answer with a 500

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url == daily_index_url(DAY):
            return httpx.Response(200, text=(self.fixtures / "form.20260911.idx").read_text())
        for accession, stem in FIXTURE_FOR.items():
            if url.endswith(f"/{accession}.txt"):
                if accession in self.failing:
                    return httpx.Response(500)
                body = (self.fixtures / f"{stem}.txt").read_text()
                if accession in self.broken:
                    body = body.replace("ownershipDocument", "somethingElse")
                return httpx.Response(200, text=body)
        return httpx.Response(404)

    @property
    def filing_requests(self) -> list[str]:
        return [u for u in self.requests if u.endswith(".txt")]


@pytest.fixture
def edgar(fixtures: Path) -> FixtureEdgar:
    return FixtureEdgar(fixtures)


@pytest.fixture
def client(edgar: FixtureEdgar, tmp_path: Path) -> EdgarClient:
    config = ClientConfig(
        user_agent=UA, backoff_base=0.0, backoff_cap=0.0, cache_root=tmp_path / "raw"
    )
    return EdgarClient(config, transport=httpx.MockTransport(edgar), sleep=lambda _: None)


@pytest.fixture
def day_lines(fixtures: Path) -> list[IndexEntry]:
    text = (fixtures / "form.20260911.idx").read_text()
    return [e for e in parse_form_idx(text) if e.form_type in {"3", "4", "4/A"}]


# --------------------------------------------------------------- grouping


def test_groups_nineteen_lines_into_five_filings(day_lines: list[IndexEntry]) -> None:
    groups = group_by_accession(day_lines)
    assert len(groups) == 5
    assert sum(len(v) for v in groups.values()) == 19
    assert len(groups[CHIME]) == 11
    assert {e.cik for e in groups[CHIME]} == {
        1795586, 1550172, 1745295, 1770474, 1782400, 1941907,
        1941908, 2072426, 2072428, 2072429, 2072430,
    }  # fmt: skip


def test_groups_keep_first_seen_order(day_lines: list[IndexEntry]) -> None:
    groups = group_by_accession(day_lines)
    assert list(groups) == list(dict.fromkeys(e.accession for e in day_lines))


# ------------------------------------------------------ one accession


def test_process_fetches_once_and_records_all_filers(
    client: EdgarClient, edgar: FixtureEdgar, day_lines: list[IndexEntry]
) -> None:
    store = MemoryStore()
    lines = group_by_accession(day_lines)[CHIME]
    with client:
        assert process_accession(client, store, CHIME, lines) is Outcome.STORED
    assert len(edgar.filing_requests) == 1

    entry = store.ledger(CHIME)
    assert entry is not None
    assert entry.outcome is Outcome.STORED
    assert entry.date_filed == DAY
    assert entry.filer_ciks == {e.cik for e in lines}
    filing = store.filing(CHIME)
    assert filing is not None
    assert filing.accession == CHIME
    assert len(filing.owners) == 10
    # The issuer is a filer but not an owner; every owner is a filer.
    assert {o.cik for o in filing.owners} < entry.filer_ciks
    assert filing.issuer_cik in entry.filer_ciks


def test_process_skips_what_the_ledger_already_has(
    client: EdgarClient, edgar: FixtureEdgar, day_lines: list[IndexEntry]
) -> None:
    store = MemoryStore()
    lines = group_by_accession(day_lines)[NWPX]
    with client:
        assert process_accession(client, store, NWPX, lines) is Outcome.STORED
        assert process_accession(client, store, NWPX, lines) is Outcome.SKIPPED
    assert len(edgar.filing_requests) == 1


def test_process_records_a_submission_with_no_xml(
    client: EdgarClient, edgar: FixtureEdgar, day_lines: list[IndexEntry]
) -> None:
    edgar.broken.add(NWPX)
    store = MemoryStore()
    lines = group_by_accession(day_lines)[NWPX]
    with client:
        assert process_accession(client, store, NWPX, lines) is Outcome.NO_XML
    entry = store.ledger(NWPX)
    assert entry is not None and entry.outcome is Outcome.NO_XML
    assert store.filing(NWPX) is None


def test_process_lets_a_server_error_propagate(
    client: EdgarClient, edgar: FixtureEdgar, day_lines: list[IndexEntry]
) -> None:
    """A 503 that survived the retries is a reason to stop, not a fact about the filing."""
    edgar.failing.add(NWPX)
    store = MemoryStore()
    lines = group_by_accession(day_lines)[NWPX]
    with client, pytest.raises(EdgarHTTPError):
        process_accession(client, store, NWPX, lines)
    assert store.ledger(NWPX) is None


# --------------------------------------------------------------- the run


def test_a_day_is_ingested_with_one_fetch_per_filing(
    client: EdgarClient, edgar: FixtureEdgar
) -> None:
    store = MemoryStore()
    with client:
        report = ingest_window(client, store, DAY, DAY, today=date(2026, 9, 15))
    assert report.counts == {Outcome.STORED: 5}
    assert len(edgar.filing_requests) == 5
    assert len(store.filings) == 5
    assert report.days_completed == [DAY]
    assert store.day_done(DAY)


def test_a_rerun_touches_nothing(client: EdgarClient, edgar: FixtureEdgar) -> None:
    store = MemoryStore()
    with client:
        ingest_window(client, store, DAY, DAY, today=date(2026, 9, 15))
        before = len(edgar.requests)
        report = ingest_window(client, store, DAY, DAY, today=date(2026, 9, 15))
    assert len(edgar.requests) == before
    assert report.counts == {}
    assert report.days_already_done == [DAY]


def test_today_is_never_marked_done(client: EdgarClient, edgar: FixtureEdgar) -> None:
    store = MemoryStore()
    with client:
        first = ingest_window(client, store, DAY, DAY, today=DAY)
        second = ingest_window(client, store, DAY, DAY, today=DAY)
    assert first.days_completed == []
    assert not store.day_done(DAY)
    # The index is re-read, but the five filings are skipped via the ledger.
    assert second.counts == {Outcome.SKIPPED: 5}
    assert len(edgar.filing_requests) == 5


def test_catch_up_over_a_window_with_missing_days(client: EdgarClient, edgar: FixtureEdgar) -> None:
    """Weekends 404 and are marked done; the one real day is ingested."""
    store = MemoryStore()
    with client:
        report = ingest_window(
            client, store, date(2026, 9, 9), date(2026, 9, 13), today=date(2026, 9, 15)
        )
    assert report.counts == {Outcome.STORED: 5}
    assert len(report.days_completed) == 5
    assert all(store.day_done(date(2026, 9, d)) for d in range(9, 14))


def test_a_bad_filing_does_not_stop_the_run(client: EdgarClient, edgar: FixtureEdgar) -> None:
    edgar.broken.add(NWPX)
    store = MemoryStore()
    with client:
        report = ingest_window(client, store, DAY, DAY, today=date(2026, 9, 15))
    assert report.counts == {Outcome.STORED: 4, Outcome.NO_XML: 1}
    assert store.day_done(DAY)


def test_a_server_error_stops_the_run_and_leaves_the_day_open(
    client: EdgarClient, edgar: FixtureEdgar
) -> None:
    edgar.failing.add(CHIME)
    store = MemoryStore()
    with client, pytest.raises(EdgarHTTPError):
        ingest_window(client, store, DAY, DAY, today=date(2026, 9, 15))
    assert not store.day_done(DAY)
