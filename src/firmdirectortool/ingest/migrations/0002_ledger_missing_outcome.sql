-- 0002. A fourth ledger outcome: 'missing'.
--
-- A filing the daily index lists but the archive no longer has (404): the
-- SEC removed it after the index was written. First seen for
-- 0001891061-26-000004 on 4 Feb 2026. Recording it lets the day be marked
-- done while still saying what happened to that filing.

ALTER TABLE ledger DROP CONSTRAINT ledger_outcome_check;
ALTER TABLE ledger ADD CONSTRAINT ledger_outcome_check
    CHECK (outcome IN ('stored', 'no_xml', 'parse_error', 'missing'));
