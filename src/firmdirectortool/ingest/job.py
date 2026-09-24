"""The ingestion job.

Walk a window of daily indexes, deduplicate the lines by accession, fetch each
filing once, parse it, and record the result. Three functions, each testable
on its own:

* :func:`group_by_accession` — index lines to one group per filing.
* :func:`process_accession` — one filing: ledger check, fetch, parse, record.
* :func:`ingest_window` — the loop over days that calls the other two.

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
from datetime import date, timedelta

from firmdirectortool.edgar import (
    OWNERSHIP_FORMS,
    EdgarClient,
    IndexEntry,
    ParseError,
    extract_ownership_xml,
    parse_ownership_document,
    walk_daily,
)

from .store import FilingStore, LedgerEntry, Outcome

__all__ = ["RunReport", "group_by_accession", "ingest_window", "process_accession"]


@dataclass
class RunReport:
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
    groups: dict[str, list[IndexEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.accession, []).append(entry)
    return groups


def process_accession(
    client: EdgarClient,
    store: FilingStore,
    accession: str,
    lines: list[IndexEntry],
) -> Outcome:

    ledgerentry = store.ledger(accession)
    if ledgerentry is not None:
        return Outcome.SKIPPED

    text = client.get_text(lines[0].url)
    try:
        xml = extract_ownership_xml(text, accession=accession)
    except ParseError:
        entry = LedgerEntry(
            accession=accession,
            outcome=Outcome.NO_XML,
            date_filed=lines[0].date_filed,
            filer_ciks=frozenset(line.cik for line in lines),
        )
        store.record(entry, None)
        return Outcome.NO_XML
    try:
        filing = parse_ownership_document(xml, accession=accession)
    except ParseError as exc:
        entry = LedgerEntry(
            accession=accession,
            outcome=Outcome.PARSE_ERROR,
            date_filed=lines[0].date_filed,
            filer_ciks=frozenset(line.cik for line in lines),
            reason=str(exc),
        )
        store.record(entry, None)
        return Outcome.PARSE_ERROR
    entry = LedgerEntry(
        accession=accession,
        outcome=Outcome.STORED,
        date_filed=lines[0].date_filed,
        filer_ciks=frozenset(line.cik for line in lines),
    )
    store.record(entry, filing)
    return Outcome.STORED


def ingest_window(
    client: EdgarClient,
    store: FilingStore,
    start: date,
    end: date,
    *,
    form_types: Iterable[str] = OWNERSHIP_FORMS,
    today: date | None = None,
) -> RunReport:

    # Resolve ``today`` once and pass the same value to walk_daily, so both
    # functions agree on which day is "today".
    today = date.today() if today is None else today
    report = RunReport(start, end)
    day = start
    while day <= end:
        if store.day_done(day):
            report.days_already_done.append(day)
        else:
            # A one-day range. walk_daily fetches the index and treats a 404 as
            # a weekend, yielding nothing; that day is then marked done below.
            lines = walk_daily(client, day, day, form_types=form_types, today=today)
            for accession, group in group_by_accession(lines).items():
                outcome = process_accession(client, store, accession, group)
                report.counts[outcome] += 1
            # Mark only after every filing in the day is recorded. A crash above
            # leaves the day open; the next run redoes it and the ledger skips
            # whatever was already stored.
            if day != today:
                store.mark_day_done(day)
                report.days_completed.append(day)
        day += timedelta(days=1)
    return report
