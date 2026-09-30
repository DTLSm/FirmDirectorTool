from dataclasses import replace
from datetime import date

import psycopg
import pytest

from firmdirectortool.edgar import parse_ownership_document
from firmdirectortool.ingest.postgres_store import PostgresStore
from firmdirectortool.ingest.store import LedgerEntry, Outcome


def test_a_day_is_done_only_once_marked(db):
    store = PostgresStore(db)
    assert not store.day_done(date(2024, 9, 11))
    store.mark_day_done(date(2024, 9, 11))
    assert store.day_done(date(2024, 9, 11))
    assert not store.day_done(date(2024, 9, 12))


def test_marking_a_day_twice_is_harmless(db):
    store = PostgresStore(db)
    store.mark_day_done(date(2024, 9, 11))
    store.mark_day_done(date(2024, 9, 11))
    assert store.day_done(date(2024, 9, 11))


def test_an_unseen_accession_has_no_ledger_entry(db):
    assert PostgresStore(db).ledger("0001234567-24-000001") is None


def test_a_ledger_row_comes_back_as_a_ledger_entry(db):
    db.execute(
        "INSERT INTO ledger (accession, outcome, date_filed, filer_ciks, reason) "
        "VALUES (%s, %s, %s, %s, %s)",
        ("0001234567-24-000001", "parse_error", date(2024, 9, 11), [320193, 1214156], "no issuer"),
    )
    assert PostgresStore(db).ledger("0001234567-24-000001") == LedgerEntry(
        accession="0001234567-24-000001",
        outcome=Outcome.PARSE_ERROR,
        date_filed=date(2024, 9, 11),
        filer_ciks=frozenset({320193, 1214156}),
        reason="no issuer",
    )


CHIME = "0001234567-24-000002"


def chime(fixtures):
    xml = (fixtures / "form4_chime_multi_owner.xml").read_text()
    return parse_ownership_document(xml)


def stored_entry(accession):
    return LedgerEntry(accession, Outcome.STORED, date(2024, 9, 11), frozenset({1795587}))


def count(db, table):
    return db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_a_recorded_entry_comes_back_from_the_ledger(db):
    store = PostgresStore(db)
    entry = LedgerEntry(CHIME, Outcome.NO_XML, date(2024, 9, 11), frozenset({1795587, 42}))
    store.record(entry, None)
    assert store.ledger(CHIME) == entry


def test_a_stored_filing_writes_one_row_per_owner(db, fixtures):
    PostgresStore(db).record(stored_entry(CHIME), chime(fixtures))
    assert count(db, "filings") == 1
    assert count(db, "reporting_owners") == 10


def test_recording_again_replaces_the_filing(db, fixtures):
    store = PostgresStore(db)
    store.record(stored_entry(CHIME), chime(fixtures))
    failed = LedgerEntry(CHIME, Outcome.PARSE_ERROR, date(2024, 9, 11), frozenset(), "bad")
    store.record(failed, None)
    assert store.ledger(CHIME) == failed
    assert count(db, "filings") == 0
    assert count(db, "reporting_owners") == 0


def test_a_failed_write_leaves_nothing_behind(db, fixtures):
    bad = replace(chime(fixtures), document_type="10-K")
    with pytest.raises(psycopg.errors.CheckViolation):
        PostgresStore(db).record(stored_entry(CHIME), bad)
    assert PostgresStore(db).ledger(CHIME) is None


def test_an_unstored_accession_has_no_filing(db):
    assert PostgresStore(db).filing(CHIME) is None


def test_a_stored_filing_comes_back_unchanged(db, fixtures):
    store = PostgresStore(db)
    store.record(stored_entry(CHIME), chime(fixtures))
    assert store.filing(CHIME) == chime(fixtures)
