"""Parsing Section 16 ownership documents.

Forms 3, 4 and 5 — and their amendments — share one XML schema,
``ownershipDocument``. This module extracts only what the interlock graph
needs: who the issuer is, who the reporting owners are, and in what capacity.
The transaction tables, holdings, footnotes and signature blocks are ignored.
That is a large fraction of the document and none of it says anything about
board membership.

The one field the whole project rests on is ``rptOwnerCik``. It is a persistent
identifier for a *person*, assigned by the SEC and reused across every issuer
they file for, which is what makes it possible to build a director graph from
public data without name disambiguation. Names are kept for display only; they
are inconsistently formatted and change on marriage, and nothing should ever
join on them.

Two layers here, deliberately separate:

* :func:`extract_ownership_xml` unwraps the complete submission text file. That
  wrapper is SGML-ish, not XML, and must not be fed to an XML parser.
* :func:`parse_ownership_document` reads the XML itself.

Keeping them apart means the parser can be tested against a bare XML fixture
and the unwrapper against a full submission, and either can be replaced. If
hand-rolling the XML becomes tedious, a library such as ``edgartools`` is a
reasonable substitute for :func:`parse_ownership_document` specifically — the
HTTP layer stays ours, the parsing need not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from xml.etree.ElementTree import Element

from .errors import ParseError

__all__ = [
    "OwnershipFiling",
    "ParseError",
    "ReportingOwner",
    "element_flag",
    "element_text",
    "extract_ownership_xml",
    "parse_ownership_document",
]


@dataclass(frozen=True)
class ReportingOwner:
    """One ``reportingOwner`` block: a person (or entity) and their relationship."""

    cik: int
    name: str
    is_director: bool
    is_officer: bool
    is_ten_percent_owner: bool
    is_other: bool
    officer_title: str | None

    @property
    def is_independent(self) -> bool:
        """A director who is not also an officer or a large holder.

        A proxy for non-executive, and only a proxy: it is what the filing
        supports, not what a governance researcher would code by hand. The
        README says so out loud.
        """
        return self.is_director and not self.is_officer and not self.is_ten_percent_owner


@dataclass(frozen=True)
class OwnershipFiling:
    """A parsed Form 3, 4 or 5."""

    document_type: str
    period_of_report: date | None
    issuer_cik: int
    issuer_name: str
    issuer_trading_symbol: str | None
    owners: tuple[ReportingOwner, ...]
    accession: str | None = None

    @property
    def directors(self) -> tuple[ReportingOwner, ...]:
        return tuple(o for o in self.owners if o.is_director)

    @property
    def is_amendment(self) -> bool:
        return self.document_type.endswith("/A")


def extract_ownership_xml(submission: str | bytes) -> str:
    """Pull the ``ownershipDocument`` XML out of a complete submission text file.

    A submission file looks like this, and the tags are *not* closed properly —
    it is dissemination-format SGML, and every XML parser chokes on it::

        <SEC-DOCUMENT>0000320193-26-000008.txt : 20260912
        <SEC-HEADER>...
        <DOCUMENT>
        <TYPE>4
        <SEQUENCE>1
        <FILENAME>wf-form4.xml
        <TEXT>
        <XML>
        <?xml version="1.0"?>
        <ownershipDocument>...</ownershipDocument>
        </XML>
        </TEXT>
        </DOCUMENT>
        <DOCUMENT>
        <TYPE>EX-24
        ...

    So: scan for the ``<XML>`` ... ``</XML>`` blocks and return the first whose
    body contains ``<ownershipDocument``. Not simply the first ``<XML>`` block —
    a filing can carry exhibits, and a power-of-attorney exhibit is not the
    form.

    Raise :class:`ParseError` if there is none. That is the expected outcome for
    filings old enough to predate mandatory XML (roughly, before mid-2003), so
    the caller should be able to count them rather than crash: make the message
    say which accession, if it was given one.
    """
    raise NotImplementedError


def element_text(parent: Element, path: str) -> str | None:
    """Text at ``path`` under ``parent``, stripped, or ``None`` if absent or blank.

    The complication: many fields in this schema are footnote-able, which means
    the value may be wrapped one level deeper::

        <officerTitle>CFO</officerTitle>
        <officerTitle><value>CFO</value><footnoteId id="F1"/></officerTitle>

    Both forms are legal and both occur. Look for a ``value`` child first, fall
    back to the element's own text. Return ``None`` for an empty string so that
    callers can use ``or`` chains without treating ``""`` as a real title.
    """
    raise NotImplementedError


def element_flag(parent: Element, path: str) -> bool:
    """A relationship flag as a bool.

    Values seen in the wild: ``1``, ``0``, ``true``, ``false``, ``Y``, ``N``,
    with surrounding whitespace, and the same ``<value>`` wrapper as
    :func:`element_text`. An absent element means false — a filer who is not an
    officer frequently just omits ``isOfficer``.

    Be strict about what counts as true, and raise :class:`ParseError` on a
    value you do not recognise rather than defaulting to false. A silently
    false ``isDirector`` deletes a person from a board.
    """
    raise NotImplementedError


def parse_ownership_document(xml: str | bytes, *, accession: str | None = None) -> OwnershipFiling:
    """Parse ``ownershipDocument`` XML into an :class:`OwnershipFiling`.

    Fields wanted, by path from the root:

    ===============================================  ===================================
    ``documentType``                                 ``3``, ``4``, ``5`` or an amendment
    ``periodOfReport``                               ISO date; may be absent
    ``issuer/issuerCik``                             zero-padded string
    ``issuer/issuerName``
    ``issuer/issuerTradingSymbol``                   often ``NONE`` or blank
    ``reportingOwner`` (repeating)
    ``  reportingOwnerId/rptOwnerCik``
    ``  reportingOwnerId/rptOwnerName``
    ``  reportingOwnerRelationship/isDirector``      and ``isOfficer``,
                                                     ``isTenPercentOwner``, ``isOther``
    ``  reportingOwnerRelationship/officerTitle``    free text
    ===============================================  ===================================

    Notes that will cost you an afternoon each if you skip them:

    * **A filing can have several reporting owners.** A joint filing by a
      director and their family trust is one accession, two owners. Model it as
      a sequence; never take ``find`` and move on.
    * The root element has no XML namespace, so plain ``find("issuer/issuerCik")``
      works. Do not add namespace handling that is not needed.
    * CIKs are zero-padded strings in the XML and integers everywhere else in
      this codebase. Convert once, here.
    * ``issuerTradingSymbol`` is frequently the literal string ``NONE``, or a
      single space. Normalise to ``None``.
    * ``officerTitle`` is free text and will be used for the ``has_ceo_exp`` and
      ``has_cfo_exp`` features later. Do **not** try to normalise it here — keep
      the raw string and do the regex work in the feature layer, where it can be
      reviewed and changed without re-parsing everything.
    * A missing or non-integer ``issuerCik``, or a ``reportingOwner`` with no
      ``rptOwnerCik``, is a corrupt document: raise :class:`ParseError` naming
      the accession. Everything else may be absent.

    The fixtures in ``tests/fixtures/edgar/`` are real documents, and they are
    the specification: where this docstring and a fixture disagree, the fixture
    is right.
    """
    raise NotImplementedError
