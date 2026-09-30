"""Run the ingestion job from the command line.

    uv run --env-file .env python -m firmdirectortool.ingest [--start DATE] [--end DATE]

With no dates, the window is the 14 days before today.
A day already done costs one query, so the wide window is nearly free, and
a run catches up by itself after up to two weeks of missed nights.
"""

import argparse
import os
from collections.abc import Sequence
from datetime import date, timedelta

import psycopg

from firmdirectortool.edgar import ClientConfig, EdgarClient

from .job import ingest_window
from .migrate import migrate
from .postgres_store import PostgresStore

#: Days in the default window, ending yesterday.
DEFAULT_WINDOW_DAYS = 14


def default_window(today: date) -> tuple[date, date]:
    """The ``DEFAULT_WINDOW_DAYS`` days before ``today``, which is still being filed."""
    end = today - timedelta(days=1)
    return end - timedelta(days=DEFAULT_WINDOW_DAYS - 1), end


def parse_window(argv: Sequence[str] | None, today: date) -> tuple[date, date]:
    """The days to ingest: those given on the command line, else the default window."""
    parser = argparse.ArgumentParser(
        prog="python -m firmdirectortool.ingest",
        description="Fetch Forms 3, 4 and 5 from EDGAR into Postgres.",
    )
    parser.add_argument("--start", type=date.fromisoformat, help="first day, YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, help="last day, YYYY-MM-DD")
    args = parser.parse_args(argv)

    default_start, default_end = default_window(today)
    start = default_start if args.start is None else args.start
    end = default_end if args.end is None else args.end
    if start > end:
        parser.error(f"--start {start} is after --end {end}")
    return start, end


def main(argv: Sequence[str] | None = None) -> int:
    """Ingest one window of days and print the report. Returns the exit code."""
    start, end = parse_window(argv, date.today())
    config = ClientConfig.from_env()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is not set; see .env.example")

    with (
        psycopg.connect(database_url, autocommit=True) as conn,
        EdgarClient(config) as client,
    ):
        migrate(conn)
        store = PostgresStore(conn)
        report = ingest_window(client, store, start, end)

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
