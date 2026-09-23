"""The ingestion job.

Walk a window of daily indexes, deduplicate the lines by accession, fetch each
filing once, parse it, and record the result. Three functions, each testable
on its own:

* :func:`group_by_accession` — index lines to one group per filing.
* :func:`process_accession` — one filing: ledger check, fetch, parse, record.
* :func:`ingest_window` — the loop over days that calls the other two.

Two promises the README makes are kept here and nowhere else:

**Idempotent.** Running the same window twice does the same thing once. The
ledger is what makes that true: an accession already recorded is skipped
before anything is fetched, and a day already marked done is skipped before
its index is even requested.

**Catch-up.** A run that missed three days is not a special case. The window
is simply wider, and days already done fall through the same skip.

What the job deliberately does *not* do:

* It does not stop on a bad filing. A filing that cannot be parsed is recorded
  as such and the run carries on; the report says how many there were, and the
  caller decides whether that number is alarming. One malformed document in
  two thousand must not lose the other 1,999 — but see the open question in
  :func:`ingest_window`.
* It does not decide what an amendment means. A ``4/A`` is stored as a
  filing with ``is_amendment`` set. Whether it supersedes the original is a
  question about board membership, and belongs in the graph layer.
* It does not filter on directors. A filing with no directors still says who
  the officers are, and ``has_ceo_exp`` in the feature layer needs that.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

from firmdirectortool.edgar import OWNERSHIP_FORMS, EdgarClient, IndexEntry

from .store import FilingStore, Outcome

__all__ = ["RunReport", "group_by_accession", "ingest_window", "process_accession"]


@dataclass
class RunReport:
    """What one call to :func:`ingest_window` did."""

    start: date
    end: date
    counts: Counter[Outcome] = field(default_factory=Counter)
    #: Days newly marked done by this run. Today is never among them.
    days_completed: list[date] = field(default_factory=list)
    #: Days skipped because an earlier run had already marked them done.
    days_already_done: list[date] = field(default_factory=list)

    def __str__(self) -> str:
        parts = ", ".join(f"{n} {outcome}" for outcome, n in sorted(self.counts.items()))
        return (
            f"{self.start} to {self.end}: {parts or 'nothing'}; "
            f"{len(self.days_completed)} days completed, "
            f"{len(self.days_already_done)} already done"
        )


def group_by_accession(entries: Iterable[IndexEntry]) -> dict[str, list[IndexEntry]]:
    """One group per filing, in first-seen order.

    The index lists a filing once per filer, and the daily index is sorted by
    form type then company name, so the eleven lines of the Chime Form 4 are
    *not* adjacent — the issuer sorts under C and the owners under their own
    names. Grouping therefore has to see the whole day before it can be sure a
    group is complete. That is fine: a day of Section 16 filings is a few
    thousand lines, and the walker already fetched the file whole.

    Do not lose lines. Every line's CIK ends up in the group, because the
    ledger keeps them all as ``filer_ciks``.
    """
    raise NotImplementedError


def process_accession(
    client: EdgarClient,
    store: FilingStore,
    accession: str,
    lines: list[IndexEntry],
) -> Outcome:
    """Fetch, parse and record one filing. Returns what happened.

    In order:

    1. If ``store.ledger(accession)`` already has an entry, return
       :attr:`~.store.Outcome.SKIPPED` without fetching. This is the
       idempotency check, and it runs before the client is touched so a rerun
       costs no request budget at all.
    2. Fetch ``lines[0].url``. All the lines name the same document under
       different filers' directories, so any one of them will do; the first is
       the one the cache is most likely to hold.
    3. :func:`~firmdirectortool.edgar.extract_ownership_xml`. A
       :class:`~firmdirectortool.edgar.ParseError` here means no XML — record
       :attr:`~.store.Outcome.NO_XML` and return.
    4. :func:`~firmdirectortool.edgar.parse_ownership_document`, passing the
       accession so its errors name the filing. A ``ParseError`` here is
       :attr:`~.store.Outcome.PARSE_ERROR`; keep the message as the entry's
       ``reason``.
    5. ``store.record`` the entry and the filing together.

    The ``filer_ciks`` on the ledger entry are the CIKs of *all* the lines,
    and ``date_filed`` comes from the index, not the document — the document
    carries a period of report, which is a different date.

    Anything the client raises other than a parse failure — a 503 that
    survived its retries, a block, a dropped connection — propagates. Those
    are not facts about the filing; they are reasons the run should stop and
    be rerun.
    """
    raise NotImplementedError


def ingest_window(
    client: EdgarClient,
    store: FilingStore,
    start: date,
    end: date,
    *,
    form_types: Iterable[str] = OWNERSHIP_FORMS,
    today: date | None = None,
) -> RunReport:
    """Ingest every ``form_types`` filing filed from ``start`` to ``end``, inclusive.

    Day by day, so that days can be marked done individually:

    1. ``store.day_done(day)`` — if so, note it on the report and move on
       without touching the network.
    2. Walk that one day with :func:`~firmdirectortool.edgar.walk_daily`,
       passing ``today`` through so the cache rule holds.
    3. Group, then :func:`process_accession` each group, tallying outcomes.
    4. ``store.mark_day_done(day)`` — **unless** ``day`` is today. Today's index
       is still growing; marking it done would make tomorrow's run skip
       whatever was filed after this run looked.

    ``today`` defaults to the real date, as in :func:`walk_daily`.

    Open question, deliberately left for the caller to answer: should a run
    with parse errors exit non-zero? The report carries the count either way.
    The CronJob wrapper in Slice 3 is the right place to pick a threshold,
    because that is where "fail the job" has a meaning.
    """
    raise NotImplementedError
