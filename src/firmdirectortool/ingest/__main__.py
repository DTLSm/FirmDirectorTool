"""Run the ingestion job from the command line.

    uv run --env-file .env python -m firmdirectortool.ingest [--start DATE] [--end DATE]
    uv run --env-file .env python -m firmdirectortool.ingest --retry-errors

With no dates, the window is the 14 days before today.
A day already done costs one query, so the wide window is nearly free, and
a run catches up by itself after up to two weeks of missed nights.

``--retry-errors`` instead processes every recorded parse error again: run it
after fixing the parser.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import psycopg

from firmdirectortool.edgar import ClientConfig, EdgarClient

from .job import ingest_window, retry_errors
from .migrate import migrate
from .postgres_store import PostgresStore

#: Days in the default window, ending yesterday.
DEFAULT_WINDOW_DAYS = 14


@dataclass(frozen=True)
class Options:
    """What the command line asked for."""

    start: date
    end: date
    retry_errors: bool


def default_window(today: date) -> tuple[date, date]:
    """The ``DEFAULT_WINDOW_DAYS`` days before ``today``, which is still being filed."""
    end = today - timedelta(days=1)
    return end - timedelta(days=DEFAULT_WINDOW_DAYS - 1), end


def parse_args(argv: Sequence[str] | None, today: date) -> Options:
    """Read the command line. Missing dates take the default window."""
    parser = argparse.ArgumentParser(
        prog="python -m firmdirectortool.ingest",
        description="Fetch Forms 3, 4 and 5 from EDGAR into Postgres.",
    )
    parser.add_argument("--start", type=date.fromisoformat, help="first day, YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, help="last day, YYYY-MM-DD")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="process every recorded parse error again, instead of a window of days",
    )
    args = parser.parse_args(argv)

    if args.retry_errors and (args.start is not None or args.end is not None):
        parser.error("--retry-errors takes no dates: it retries every parse error")
    default_start, default_end = default_window(today)
    start = default_start if args.start is None else args.start
    end = default_end if args.end is None else args.end
    if start > end:
        parser.error(f"--start {start} is after --end {end}")
    return Options(start, end, args.retry_errors)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the job once and print the report. Returns the exit code."""
    options = parse_args(argv, date.today())
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
        if options.retry_errors:
            report = retry_errors(client, store)
        else:
            report = ingest_window(client, store, options.start, options.end)

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
