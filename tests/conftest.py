import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest

from firmdirectortool.ingest.migrate import migrate


@pytest.fixture(scope="session")
def fixtures() -> Path:
    """Directory of real EDGAR documents. See its README for provenance."""
    return Path(__file__).parent / "fixtures" / "edgar"


@pytest.fixture(scope="session")
def database_url() -> str:
    """The test database. Tests that need it skip when TEST_DATABASE_URL is unset."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set; start Postgres with docker compose up -d")
    return url


@pytest.fixture(scope="session")
def migrated_url(database_url: str) -> str:
    """The same URL, after the migrations have been applied. Runs once per pytest run."""
    with psycopg.connect(database_url, autocommit=True) as conn:
        migrate(conn)
    return database_url


@pytest.fixture
def db(migrated_url: str) -> Iterator[psycopg.Connection[Any]]:
    """A connection to a migrated database with every table emptied."""
    with psycopg.connect(migrated_url, autocommit=True) as conn:
        conn.execute("TRUNCATE filings, reporting_owners, ledger, days_done")
        yield conn
