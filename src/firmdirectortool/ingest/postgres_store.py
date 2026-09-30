"""The :class:`~firmdirectortool.ingest.store.FilingStore` in Postgres.

The tables come from the migrations in ``migrations/``; this module only reads
and writes rows. It needs a connection opened with ``autocommit=True``: every
statement outside a ``conn.transaction()`` block is saved at once, and
:meth:`PostgresStore.record` groups its writes into one transaction itself.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import psycopg

from firmdirectortool.edgar import OwnershipFiling, ReportingOwner

from .store import LedgerEntry, Outcome

__all__ = ["PostgresStore"]


class PostgresStore:
    """A :class:`~firmdirectortool.ingest.store.FilingStore` over one connection."""

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        if not conn.autocommit:
            raise ValueError("PostgresStore needs a connection opened with autocommit=True")
        self._conn = conn

    def ledger(self, accession: str) -> LedgerEntry | None:
        row = self._conn.execute(
            "SELECT accession, outcome, date_filed, filer_ciks, reason "
            "FROM ledger WHERE accession = %s",
            (accession,),
        ).fetchone()
        if row is None:
            return None
        return _ledger_entry(row)

    def entries_with(self, outcome: Outcome) -> list[LedgerEntry]:
        rows = self._conn.execute(
            "SELECT accession, outcome, date_filed, filer_ciks, reason "
            "FROM ledger WHERE outcome = %s ORDER BY date_filed, accession",
            (outcome.value,),
        ).fetchall()
        return [_ledger_entry(row) for row in rows]

    def record(self, entry: LedgerEntry, filing: OwnershipFiling | None) -> None:
        if (filing is None) != (entry.outcome is not Outcome.STORED):
            raise ValueError(f"outcome {entry.outcome} and filing={filing!r} disagree")
        with self._conn.transaction():
            self._conn.execute("DELETE FROM ledger WHERE accession = %s", (entry.accession,))
            self._conn.execute("DELETE FROM filings WHERE accession = %s", (entry.accession,))
            self._conn.execute(
                "INSERT INTO ledger (accession, outcome, date_filed, filer_ciks, reason) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    entry.accession,
                    entry.outcome.value,
                    entry.date_filed,
                    sorted(entry.filer_ciks),
                    entry.reason,
                ),
            )
            if filing is not None:
                self._conn.execute(
                    "INSERT INTO filings (accession, document_type, date_filed, period_of_report, "
                    "issuer_cik, issuer_name, issuer_trading_symbol) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (
                        entry.accession,
                        filing.document_type,
                        entry.date_filed,
                        filing.period_of_report,
                        filing.issuer_cik,
                        filing.issuer_name,
                        filing.issuer_trading_symbol,
                    ),
                )
                for position, owner in enumerate(filing.owners):
                    self._conn.execute(
                        "INSERT INTO reporting_owners (accession, position, cik, name, "
                        "is_director, is_officer, is_ten_percent_owner, is_other, officer_title) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            entry.accession,
                            position,
                            owner.cik,
                            owner.name,
                            owner.is_director,
                            owner.is_officer,
                            owner.is_ten_percent_owner,
                            owner.is_other,
                            owner.officer_title,
                        ),
                    )

    def filing(self, accession: str) -> OwnershipFiling | None:
        header = self._conn.execute(
            "SELECT document_type, period_of_report, issuer_cik, issuer_name, "
            "issuer_trading_symbol FROM filings WHERE accession = %s",
            (accession,),
        ).fetchone()
        if header is None:
            return None
        document_type, period_of_report, issuer_cik, issuer_name, issuer_trading_symbol = header
        rows = self._conn.execute(
            "SELECT cik, name, is_director, is_officer, is_ten_percent_owner, is_other, "
            "officer_title FROM reporting_owners WHERE accession = %s ORDER BY position",
            (accession,),
        ).fetchall()
        owners = []
        for cik, name, is_director, is_officer, is_ten_percent_owner, is_other, title in rows:
            owners.append(
                ReportingOwner(
                    cik=cik,
                    name=name,
                    is_director=is_director,
                    is_officer=is_officer,
                    is_ten_percent_owner=is_ten_percent_owner,
                    is_other=is_other,
                    officer_title=title,
                )
            )
        return OwnershipFiling(
            document_type=document_type,
            period_of_report=period_of_report,
            issuer_cik=issuer_cik,
            issuer_name=issuer_name,
            issuer_trading_symbol=issuer_trading_symbol,
            owners=tuple(owners),
            accession=accession,
        )

    def day_done(self, day: date) -> bool:
        row = self._conn.execute("SELECT 1 FROM days_done WHERE day = %s", (day,)).fetchone()
        return row is not None

    def mark_day_done(self, day: date) -> None:
        self._conn.execute(
            "INSERT INTO days_done (day) VALUES (%s) ON CONFLICT (day) DO NOTHING", (day,)
        )


def _ledger_entry(row: tuple[Any, ...]) -> LedgerEntry:
    """A ``ledger`` row, selected as accession, outcome, date_filed, filer_ciks, reason."""
    accession, outcome, date_filed, filer_ciks, reason = row
    return LedgerEntry(
        accession=accession,
        outcome=Outcome(outcome),
        date_filed=date_filed,
        filer_ciks=frozenset(filer_ciks),
        reason=reason,
    )
