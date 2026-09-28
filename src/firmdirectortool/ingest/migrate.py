"""Apply the numbered SQL files in ``migrations/``, in order, once each.

The runner owns one table, ``schema_migrations``, holding the name of every
file it has applied. A file is applied inside a transaction together with the
row that records it, so a failed migration leaves neither half behind and the
next run retries it from scratch.

Files are never edited once applied. A change to the schema is a new file with
the next number; the runner does not compare contents, only names.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

import psycopg

__all__ = ["applied", "migrate", "migration_names"]

_MIGRATIONS = resources.files("firmdirectortool.ingest") / "migrations"

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    name        TEXT        PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def migration_names() -> list[str]:
    """Every ``*.sql`` file shipped with the package, in the order to apply them."""
    return sorted(p.name for p in _MIGRATIONS.iterdir() if p.name.endswith(".sql"))


def applied(conn: psycopg.Connection[Any]) -> set[str]:
    """Names already recorded in ``schema_migrations``. Creates the table if absent."""
    with conn.transaction():
        conn.execute(_BOOTSTRAP)
        rows = conn.execute("SELECT name FROM schema_migrations").fetchall()
    return {name for (name,) in rows}


def migrate(conn: psycopg.Connection[Any]) -> list[str]:
    """Apply every pending file and return the names applied, in order."""
    done = applied(conn)
    newly_applied: list[str] = []
    for name in migration_names():
        if name in done:
            continue
        sql = (_MIGRATIONS / name).read_text()
        with conn.transaction():
            conn.execute(sql)
            conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (name,))
        newly_applied.append(name)
    return newly_applied
