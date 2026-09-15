"""EDGAR index files — the feed the ingestion job walks.

EDGAR publishes what was filed, when, as plain-text index files:

* **Daily**, ``Archives/edgar/daily-index/YYYY/QTRn/form.YYYYMMDD.idx`` — one
  file per business day, sorted by form type. This is the incremental feed.
  There is no file for weekends or market holidays, and the current day's file
  grows through the day, so it is the one thing this package does not trust its
  cache for.
* **Quarterly**, ``Archives/edgar/full-index/YYYY/QTRn/form.idx`` — every
  filing in the quarter, same record layout. This is how a historical window is
  bootstrapped without making ninety daily requests. It is tens of megabytes,
  and EDGAR ignores ``Range`` requests on it.

One parser and one entry type serve both; only the URL differs. That is what
keeps bootstrap and incremental modes on a single code path, which in turn is
what makes the catch-up behaviour trustworthy: the path that backfills a year
is the path that runs every night.

**An index line is a filer, not a filing.** A Form 4 is filed jointly by the
issuer and every reporting owner, and the index lists it once *per filer*, each
line carrying that filer's CIK and a URL under that filer's directory. One
Chime Financial Form 4 in ``tests/fixtures/edgar/`` appears on eleven lines: the
issuer and ten reporting owners. All eleven URLs return the same document. So
the accession number, not the line, is the unit of work, and the ingestion job
must deduplicate on it before fetching — otherwise it spends eleven requests
and eleven parses to learn one thing.

Note also that the index gives a *filename*, not a document. For Forms 3/4/5
the file it names is the complete submission text file, which contains the
ownership XML inline — one request per filing rather than two (fetch the folder
listing, then fetch the document). See :func:`~.parse.extract_ownership_xml`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date

from .client import EdgarClient

ARCHIVES = "https://www.sec.gov/Archives"

#: Section 16 ownership forms, including amendments.
OWNERSHIP_FORMS = frozenset({"3", "3/A", "4", "4/A", "5", "5/A"})


def quarter_of(day: date) -> int:
    """1-4. EDGAR partitions both index trees by calendar quarter."""
    return (day.month - 1) // 3 + 1


def daily_index_url(day: date) -> str:
    """URL of the form-sorted daily index for ``day``.

    No check that ``day`` is a business day: the caller finds out by getting a
    404, which :func:`walk_daily` treats as "nothing filed" rather than as an
    error. Guessing the SEC's holiday calendar locally would be one more thing
    to keep correct.
    """
    return f"{ARCHIVES}/edgar/daily-index/{day.year}/QTR{quarter_of(day)}/form.{day:%Y%m%d}.idx"


def quarterly_index_url(year: int, quarter: int) -> str:
    """URL of the form-sorted full index for a quarter."""
    if not 1 <= quarter <= 4:
        raise ValueError(f"quarter must be 1-4, got {quarter}")
    return f"{ARCHIVES}/edgar/full-index/{year}/QTR{quarter}/form.idx"


@dataclass(frozen=True)
class IndexEntry:
    """One line of an index file: a filer's view of a filing, before anything is fetched."""

    form_type: str
    company_name: str
    #: The *filer's* CIK — the issuer on one line, a reporting owner on another.
    cik: int
    date_filed: date
    #: Path relative to ``Archives``, e.g. ``edgar/data/320193/0000320193-26-000008.txt``.
    filename: str

    @property
    def accession(self) -> str:
        """``0000320193-26-000008`` — the dashed form, derived from :attr:`filename`.

        This is the primary key for the whole pipeline and the deduplication
        key across index lines, so be strict: if the filename does not end in
        something shaped like an accession number (ten digits, two digits, six
        digits, dash-separated) raise :class:`~.errors.ParseError` rather than
        returning a plausible-looking wrong answer.
        """
        raise NotImplementedError

    @property
    def accession_nodash(self) -> str:
        """``000032019326000008`` — the form EDGAR uses in folder paths."""
        raise NotImplementedError

    @property
    def url(self) -> str:
        """Absolute URL of the file this entry names."""
        return f"{ARCHIVES}/{self.filename}"

    @property
    def folder_url(self) -> str:
        """Absolute URL of the accession's folder, which lists its documents."""
        return f"{ARCHIVES}/edgar/data/{self.cik}/{self.accession_nodash}"


def parse_form_idx(text: str) -> Iterator[IndexEntry]:
    """Parse a ``form.idx`` body into entries.

    Records are fixed-width, and the temptation is to read the column offsets
    off the header. Resist it — the two flavours do not agree:

    * The **quarterly** file has a single header line and a line of dashes.
    * The **daily** file wraps its header across *two* lines, splitting it
      mid-way through ``CIK   Date Filed``.
    * The line of dashes is one unbroken run in both. It marks where the
      records start; it does not encode column widths.

    A rule that works on both, and does not care if a column is widened:

    1. Strip the line, then :meth:`str.rsplit` it into four parts with
       ``maxsplit=3``. The last three fields — filename, date filed, CIK —
       contain no spaces, so this peels them off from the right exactly.
    2. What remains is form type and company name run together. Split it on the
       *first* run of two or more spaces. Form types contain single spaces
       (``1-A POS``) but never double ones; company names may contain either,
       and splitting only once means internal spacing is left alone.

    Dates come in two formats: ``YYYYMMDD`` daily, ``YYYY-MM-DD`` quarterly.
    Accept both.

    Skip the preamble, the header lines and blanks — everything before the line
    of dashes. After it, a line that does not yield five fields, or whose CIK is
    not an integer, is a corrupt record: raise :class:`~.errors.ParseError`
    naming the line number. Silently dropping records here would surface much
    later as a board with a missing director and no way to tell why.

    ``tests/fixtures/edgar/`` has a trimmed copy of each flavour.
    """
    raise NotImplementedError


def walk_daily(
    client: EdgarClient,
    start: date,
    end: date,
    *,
    form_types: Iterable[str] = OWNERSHIP_FORMS,
    today: date | None = None,
) -> Iterator[IndexEntry]:
    """Yield entries for ``form_types`` filed between ``start`` and ``end``, inclusive.

    Walks day by day. Points to get right:

    * A missing daily index (:class:`~.errors.NotFoundError`) is a weekend or a
      holiday — skip it and carry on. Any other error propagates: a 503 that
      survived the client's retries means the run should fail loudly and be
      retried later, not quietly produce a gap.
    * Pass ``use_cache=False`` for ``today``'s index. The file is still being
      appended to; a cached copy from this morning is silent data loss. Earlier
      days are immutable and should come from cache. ``today`` is a parameter
      rather than a call to :meth:`date.today` so this is testable.
    * Filter on ``form_types`` as a set membership test, not a prefix match.
      ``"4"`` must not match form ``"424B2"``. Amendments (``4/A``) are separate
      form types and are in :data:`OWNERSHIP_FORMS` on purpose — decide what to
      do with them in the parser, not here.
    * Yield lazily, and yield *every* matching line, duplicates included. This
      function reports what the index says; deduplicating by accession is the
      ingestion job's decision, and it wants the per-filer lines to cross-check
      against the reporting owners it finds in the document.
    """
    raise NotImplementedError


def walk_quarterly(
    client: EdgarClient,
    start: date,
    end: date,
    *,
    form_types: Iterable[str] = OWNERSHIP_FORMS,
) -> Iterator[IndexEntry]:
    """Same, from quarterly full indexes — for bootstrapping a historical window.

    Fetch each quarter that overlaps ``[start, end]``, then filter entries by
    ``date_filed`` so the boundaries are honoured exactly. Parse streaming
    rather than building a list: a quarter of filings is hundreds of thousands
    of records.
    """
    raise NotImplementedError
