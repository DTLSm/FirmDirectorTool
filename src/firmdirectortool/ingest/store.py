"""Where the ingestion job writes, and how it remembers what it has done.

Two kinds of record, one interface:

* **Filings.** A parsed :class:`~firmdirectortool.edgar.OwnershipFiling`, keyed
  by accession number.
* **The ledger.** One :class:`LedgerEntry` per accession the job has *seen*,
  whether or not it produced a filing, plus a per-day completion mark.

The ledger is the decision from the "store versus ledger" discussion. A store
keyed by accession can say "this filing exists"; it cannot say "this
accession was looked at and had no XML", so a pre-2003 filing would be fetched
and rejected on every run. Nor can it say "11 September was walked to the end",
so catch-up after a week of failed runs would have to re-walk the week to
discover what is missing. The ledger answers both directly, and it is what
Slice 6 monitoring will read to say "last night's run stored 1,840 filings and
skipped 3".

The interface is a :class:`typing.Protocol` rather than a base class so the
Postgres implementation in the next step and :class:`MemoryStore` here need not
share any code, only a shape. The job is written against the protocol and
tested against the in-memory store; the Postgres store gets its own tests
against a real database.

**Atomicity is the one thing the protocol demands of an implementation.**
:meth:`FilingStore.record` takes the ledger entry and the filing together,
because a crash between "write the filing" and "write the ledger entry" leaves
the two disagreeing, and the ledger is what the next run trusts. In Postgres
that is one transaction. In memory it is two dict writes and no crash can
fall between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Protocol

from firmdirectortool.edgar import OwnershipFiling


class Outcome(StrEnum):
    """What happened to one accession during one run."""

    #: Fetched, parsed, and written.
    STORED = "stored"
    #: Fetched, but the submission carried no ``ownershipDocument`` XML.
    #: Expected for filings that predate mandatory XML; counted, not fatal.
    NO_XML = "no_xml"
    #: Fetched, XML found, but it could not be understood. Worth looking at.
    PARSE_ERROR = "parse_error"
    #: Already in the ledger from an earlier run; nothing was fetched.
    SKIPPED = "skipped"


@dataclass(frozen=True)
class LedgerEntry:
    """One accession, as the job last saw it."""

    accession: str
    outcome: Outcome
    date_filed: date
    #: The CIKs of every index line that named this accession — the issuer and
    #: each reporting owner. Kept so the filers can be cross-checked against
    #: the owners the parser found; a mismatch is a filing worth a look.
    filer_ciks: frozenset[int]
    #: Free text for :attr:`Outcome.PARSE_ERROR`: the parser's message.
    reason: str | None = None


class FilingStore(Protocol):
    """The shape the ingestion job needs. See the module docstring on atomicity."""

    def ledger(self, accession: str) -> LedgerEntry | None:
        """The ledger entry for ``accession``, or ``None`` if never seen."""
        ...

    def record(self, entry: LedgerEntry, filing: OwnershipFiling | None) -> None:
        """Write the ledger entry and, when there is one, the filing — together.

        ``filing`` is ``None`` for every outcome except :attr:`Outcome.STORED`.
        Recording the same accession again replaces the earlier entry: a
        rerun after a parser fix should be able to turn a ``parse_error`` into
        a ``stored``.
        """
        ...

    def filing(self, accession: str) -> OwnershipFiling | None:
        """The stored filing, or ``None``."""
        ...

    def day_done(self, day: date) -> bool:
        """Whether every accession filed on ``day`` has been recorded."""
        ...

    def mark_day_done(self, day: date) -> None:
        """Record that ``day`` was walked to the end. Never called for today."""
        ...


class MemoryStore:
    """A :class:`FilingStore` in dicts. For tests, and for nothing else."""

    def __init__(self) -> None:
        self._ledger: dict[str, LedgerEntry] = {}
        self._filings: dict[str, OwnershipFiling] = {}
        self._days_done: set[date] = set()

    def ledger(self, accession: str) -> LedgerEntry | None:
        return self._ledger.get(accession)

    def record(self, entry: LedgerEntry, filing: OwnershipFiling | None) -> None:
        if (filing is None) != (entry.outcome is not Outcome.STORED):
            raise ValueError(f"outcome {entry.outcome} and filing={filing!r} disagree")
        self._ledger[entry.accession] = entry
        if filing is not None:
            self._filings[entry.accession] = filing
        else:
            self._filings.pop(entry.accession, None)

    def filing(self, accession: str) -> OwnershipFiling | None:
        return self._filings.get(accession)

    def day_done(self, day: date) -> bool:
        return day in self._days_done

    def mark_day_done(self, day: date) -> None:
        self._days_done.add(day)

    # Conveniences for assertions; not part of the protocol.

    @property
    def entries(self) -> list[LedgerEntry]:
        return list(self._ledger.values())

    @property
    def filings(self) -> list[OwnershipFiling]:
        return list(self._filings.values())


__all__ = ["FilingStore", "LedgerEntry", "MemoryStore", "Outcome"]
