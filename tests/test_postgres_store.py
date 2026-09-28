from datetime import date

from firmdirectortool.ingest.postgres_store import PostgresStore


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
