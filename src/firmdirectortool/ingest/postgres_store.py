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

from firmdirectortool.edgar import OwnershipFiling

from .store import LedgerEntry

__all__ = ["PostgresStore"]


class PostgresStore:
    """A :class:`~firmdirectortool.ingest.store.FilingStore` over one connection."""

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        if not conn.autocommit:
            raise ValueError("PostgresStore needs a connection opened with autocommit=True")
        self._conn = conn

    def ledger(self, accession: str) -> LedgerEntry | None:
        raise NotImplementedError

    def record(self, entry: LedgerEntry, filing: OwnershipFiling | None) -> None:
        raise NotImplementedError

    def filing(self, accession: str) -> OwnershipFiling | None:
        raise NotImplementedError

    def day_done(self, day: date) -> bool:
        row = self._conn.execute("SELECT 1 FROM days_done WHERE day = %s", (day,)).fetchone()
        return row is not None

    def mark_day_done(self, day: date) -> None:
        (
            self._conn.execute(
                "INSERT INTO days_done (day) VALUES (%s) ON CONFLICT (day) DO NOTHING", (day,)
            ),
        )
