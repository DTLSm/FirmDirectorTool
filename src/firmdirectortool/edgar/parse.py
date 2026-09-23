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

import re
from dataclasses import dataclass, field
from datetime import date
from xml.etree.ElementTree import Element, fromstring
from xml.etree.ElementTree import ParseError as XMLParseError

from .errors import ParseError

#: An ``<XML>`` block in a dissemination-format submission file. The wrapper
#: tags are upper-case and on their own lines; the body is anything, lazily.
_XML_BLOCK = re.compile(r"<XML>\s*(.*?)\s*</XML>", re.DOTALL)

_TRUE = frozenset({"1", "true", "y", "yes"})
_FALSE = frozenset({"0", "false", "n", "no"})

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
    #: Provenance, not content: two parses of the same document are equal
    #: whatever the caller labelled them, so this is excluded from ``==``.
    accession: str | None = field(default=None, compare=False)

    @property
    def directors(self) -> tuple[ReportingOwner, ...]:
        return tuple(o for o in self.owners if o.is_director)

    @property
    def is_amendment(self) -> bool:
        return self.document_type.endswith("/A")


def extract_ownership_xml(submission: str | bytes, *, accession: str | None = None) -> str:
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
    text = (
        submission.decode("utf-8", errors="replace")
        if isinstance(submission, bytes)
        else submission
    )
    for match in _XML_BLOCK.finditer(text):
        body = match.group(1)
        if "<ownershipDocument" in body:
            return body
    where = f" {accession}" if accession else ""
    raise ParseError(f"no ownershipDocument XML in submission{where}")


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
    element = parent.find(path)
    if element is None:
        return None
    value = element.find("value")
    text = (value if value is not None else element).text
    if text is None:
        return None
    text = text.strip()
    return text or None


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
    text = element_text(parent, path)
    if text is None:
        return False
    lowered = text.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ParseError(f"{path}: unrecognised flag value {text!r}")


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
    where = f" in {accession}" if accession else ""
    try:
        root = fromstring(xml)
    except XMLParseError as exc:
        raise ParseError(f"malformed ownership XML{where}: {exc}") from exc
    if root.tag != "ownershipDocument":
        raise ParseError(f"expected <ownershipDocument>{where}, got <{root.tag}>")

    document_type = element_text(root, "documentType")
    if document_type is None:
        raise ParseError(f"no documentType{where}")

    issuer_cik = _cik(element_text(root, "issuer/issuerCik"), "issuerCik", where)

    period_text = element_text(root, "periodOfReport")
    try:
        period_of_report = date.fromisoformat(period_text) if period_text else None
    except ValueError:
        raise ParseError(f"unreadable periodOfReport {period_text!r}{where}") from None

    symbol = element_text(root, "issuer/issuerTradingSymbol")
    if symbol is not None and symbol.upper() in {"NONE", "N/A"}:
        symbol = None

    owners = []
    for block in root.findall("reportingOwner"):
        owners.append(
            ReportingOwner(
                cik=_cik(element_text(block, "reportingOwnerId/rptOwnerCik"), "rptOwnerCik", where),
                name=element_text(block, "reportingOwnerId/rptOwnerName") or "",
                is_director=_flag(block, "isDirector", where),
                is_officer=_flag(block, "isOfficer", where),
                is_ten_percent_owner=_flag(block, "isTenPercentOwner", where),
                is_other=_flag(block, "isOther", where),
                officer_title=element_text(block, "reportingOwnerRelationship/officerTitle"),
            )
        )

    return OwnershipFiling(
        document_type=document_type,
        period_of_report=period_of_report,
        issuer_cik=issuer_cik,
        issuer_name=element_text(root, "issuer/issuerName") or "",
        issuer_trading_symbol=symbol,
        owners=tuple(owners),
        accession=accession,
    )


def _cik(text: str | None, field: str, where: str) -> int:
    """A zero-padded CIK string as an int, or :class:`ParseError` if missing or not a number."""
    if text is None:
        raise ParseError(f"missing {field}{where}")
    try:
        return int(text)
    except ValueError:
        raise ParseError(f"{field} {text!r} is not an integer{where}") from None


def _flag(owner: Element, name: str, where: str) -> bool:
    """:func:`element_flag` on a relationship flag, with the accession in the error."""
    try:
        return element_flag(owner, f"reportingOwnerRelationship/{name}")
    except ParseError as exc:
        raise ParseError(f"{exc}{where}") from None
