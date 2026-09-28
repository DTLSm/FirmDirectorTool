from firmdirectortool.ingest.migrate import migrate


def test_migrate_twice_applies_nothing(db):
    assert migrate(db) == []


def test_migrate_creates_the_tables(db):
    rows = db.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'").fetchall()
    names = {name for (name,) in rows}
    assert {"filings", "reporting_owners", "ledger", "days_done", "schema_migrations"} <= names
