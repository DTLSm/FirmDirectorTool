"""The submissions API — per-entity metadata.

``https://data.sec.gov/submissions/CIK##########.json`` returns an entity's
name, tickers, exchange listings, fiscal year end and filing history. The
ingestion job uses it for issuers only, and only for the handful of fields the
index files do not carry: a readable name that is not the ALL-CAPS index form,
the ticker, and the fiscal year end that anchors the annual board snapshot
(see the "what is a board-year" decision in the data model).

Different host from the archives, same rate-limit budget — which is exactly why
:class:`~.client.EdgarClient` holds one bucket rather than one per host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .client import EdgarClient

SUBMISSIONS_BASE = "https://data.sec.gov/submissions"


def submissions_url(cik: int) -> str:
    """URL for an entity's submissions document.

    The CIK is zero-padded to ten digits; the API 404s on the unpadded form.
    """
    if cik <= 0:
        raise ValueError(f"cik must be positive, got {cik}")
    return f"{SUBMISSIONS_BASE}/CIK{cik:010d}.json"


@dataclass(frozen=True)
class EntitySubmissions:
    """The subset of the submissions payload this project stores."""

    cik: int
    name: str
    tickers: tuple[str, ...]
    #: ``MMDD``, e.g. ``"0930"``. Absent for many non-operating entities.
    fiscal_year_end: str | None
    entity_type: str | None
    sic: str | None
    former_names: tuple[str, ...]


def parse_submissions(payload: Any) -> EntitySubmissions:
    """Pull :class:`EntitySubmissions` out of a decoded submissions document.

    Things the real payload will do to you:

    * ``cik`` comes back as a *string*, and not always zero-padded.
    * ``tickers`` is often ``[]`` — plenty of Section 16 issuers are not listed,
      or file under an entity that is not the listed one.
    * ``fiscalYearEnd`` is missing on some entities and is ``"--0930"`` shaped
      on others. Normalise to four digits or ``None``.
    * ``formerNames`` is a list of objects, not strings.
    * The payload is large. Take the fields, drop the rest; do not keep the
      ``filings`` block in memory when the caller only wanted a ticker.

    A missing ``name`` is a corrupt document: raise :class:`~.errors.ParseError`.
    Everything else degrades to ``None`` or an empty tuple.
    """
    raise NotImplementedError


def fetch_submissions(client: EdgarClient, cik: int) -> EntitySubmissions:
    """Fetch and parse an entity's metadata."""
    return parse_submissions(client.get_json(submissions_url(cik)))
