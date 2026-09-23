"""The submissions endpoint: URL padding and the field subset we keep.

The payload here is synthetic but shaped like the real one — the fixture
directory holds only public-domain documents fetched verbatim, and a real
submissions payload is a megabyte of filing history we do not want.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from firmdirectortool.edgar import (
    ClientConfig,
    EdgarClient,
    ParseError,
    fetch_submissions,
    parse_submissions,
    submissions_url,
)

UA = "Test Runner test@example.org"

APPLE = {
    "cik": "320193",
    "entityType": "operating",
    "sic": "3571",
    "name": "Apple Inc.",
    "tickers": ["AAPL"],
    "exchanges": ["Nasdaq"],
    "fiscalYearEnd": "0927",
    "formerNames": [
        {"name": "APPLE COMPUTER INC", "from": "1994-01-26T00:00:00.000Z", "to": "2007-01-04"},
        {"name": "APPLE COMPUTER INC/ FA", "from": "1997-07-28", "to": "1997-07-28"},
    ],
    "filings": {"recent": {"accessionNumber": ["0000320193-26-000008"] * 1000}},
}


def test_submissions_url_is_zero_padded() -> None:
    assert submissions_url(320193) == "https://data.sec.gov/submissions/CIK0000320193.json"
    with pytest.raises(ValueError):
        submissions_url(0)


def test_parses_the_fields_we_keep_and_drops_the_rest() -> None:
    entity = parse_submissions(APPLE)
    assert entity.cik == 320193
    assert entity.name == "Apple Inc."
    assert entity.tickers == ("AAPL",)
    assert entity.fiscal_year_end == "0927"
    assert entity.entity_type == "operating"
    assert entity.sic == "3571"
    assert entity.former_names == ("APPLE COMPUTER INC", "APPLE COMPUTER INC/ FA")
    assert not hasattr(entity, "filings")


def test_a_sparse_entity_degrades_to_none_and_empty() -> None:
    """A reporting owner's own CIK page: no ticker, no fiscal year, nothing much."""
    entity = parse_submissions({"cik": "0001233202", "name": "FRANSON MICHAEL C", "tickers": []})
    assert entity.cik == 1233202
    assert entity.tickers == ()
    assert entity.fiscal_year_end is None
    assert entity.entity_type is None
    assert entity.sic is None
    assert entity.former_names == ()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0930", "0930"), ("--0930", "0930"), ("", None), (None, None), ("Sept", None)],
)
def test_fiscal_year_end_is_normalised(raw: str | None, expected: str | None) -> None:
    payload = {"cik": "1", "name": "X", "fiscalYearEnd": raw}
    assert parse_submissions(payload).fiscal_year_end == expected


def test_a_missing_name_is_corrupt() -> None:
    with pytest.raises(ParseError):
        parse_submissions({"cik": "1"})
    with pytest.raises(ParseError):
        parse_submissions({"cik": "1", "name": "  "})


def test_a_missing_or_garbled_cik_is_corrupt() -> None:
    with pytest.raises(ParseError):
        parse_submissions({"name": "X"})
    with pytest.raises(ParseError):
        parse_submissions({"name": "X", "cik": "CIK0000320193"})


def test_a_non_object_payload_is_corrupt() -> None:
    with pytest.raises(ParseError):
        parse_submissions(["not", "an", "object"])


def test_fetch_uses_the_client(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=APPLE)

    config = ClientConfig(user_agent=UA, cache_root=tmp_path / "raw")
    with EdgarClient(config, transport=httpx.MockTransport(handler), sleep=lambda _: None) as c:
        assert fetch_submissions(c, 320193).name == "Apple Inc."
    assert seen == ["https://data.sec.gov/submissions/CIK0000320193.json"]
