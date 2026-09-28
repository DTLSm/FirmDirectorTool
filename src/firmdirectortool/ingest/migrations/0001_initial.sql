-- 0001. The four tables the ingestion job writes.
--
-- Two kinds of record, as in store.py: the filings (a header row plus one
-- row per reporting owner) and the ledger (one row per accession seen, plus
-- one row per day walked to the end). Nothing here is derived; issuers and
-- persons are views over these tables, built when the graph layer needs them.

CREATE TABLE filings (
    accession               TEXT        PRIMARY KEY
                                        CHECK (accession ~ '^[0-9]{10}-[0-9]{2}-[0-9]{6}$'),
    document_type           TEXT        NOT NULL
                                        CHECK (document_type IN ('3', '4', '5', '3/A', '4/A', '5/A')),
    date_filed              DATE        NOT NULL,
    period_of_report        DATE,
    issuer_cik              BIGINT      NOT NULL,
    issuer_name             TEXT        NOT NULL,
    issuer_trading_symbol   TEXT
);

CREATE INDEX filings_issuer_cik_idx ON filings (issuer_cik);
CREATE INDEX filings_date_filed_idx ON filings (date_filed);

CREATE TABLE reporting_owners (
    accession               TEXT        NOT NULL REFERENCES filings ON DELETE CASCADE,
    cik                     BIGINT      NOT NULL,
    name                    TEXT        NOT NULL,
    is_director             BOOLEAN     NOT NULL,
    is_officer              BOOLEAN     NOT NULL,
    is_ten_percent_owner    BOOLEAN     NOT NULL,
    is_other                BOOLEAN     NOT NULL,
    officer_title           TEXT,
    PRIMARY KEY (accession, cik)
);

CREATE INDEX reporting_owners_cik_idx ON reporting_owners (cik);

CREATE TABLE ledger (
    accession               TEXT        PRIMARY KEY
                                        CHECK (accession ~ '^[0-9]{10}-[0-9]{2}-[0-9]{6}$'),
    -- 'skipped' is deliberately absent: the job never records a skip, it
    -- returns before the store is touched. A row claiming it is a bug.
    outcome                 TEXT        NOT NULL
                                        CHECK (outcome IN ('stored', 'no_xml', 'parse_error')),
    date_filed              DATE        NOT NULL,
    filer_ciks              BIGINT[]    NOT NULL,
    reason                  TEXT        CHECK (reason IS NULL OR outcome = 'parse_error'),
    recorded_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ledger_date_filed_idx ON ledger (date_filed);

CREATE TABLE days_done (
    day                     DATE        PRIMARY KEY,
    marked_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);
