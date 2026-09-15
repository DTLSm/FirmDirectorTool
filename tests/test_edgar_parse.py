"""Parser tests, all against real documents.

Every expectation below was read off the fixture, not invented. If one of these
fails after a change, the change is wrong — or EDGAR is, which is worth knowing
either way.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from xml.etree.ElementTree import fromstring

import pytest

from firmdirectortool.edgar import (
    OwnershipFiling,
    ParseError,
    extract_ownership_xml,
    parse_ownership_document,
)
from firmdirectortool.edgar.parse import element_flag, element_text


def load(fixtures: Path, stem: str) -> OwnershipFiling:
    return parse_ownership_document((fixtures / f"{stem}.xml").read_text(), accession=stem)


# ------------------------------------------------------- unwrapping the SGML


@pytest.mark.parametrize(
    "stem",
    [
        "form3_ambev",
        "form4_union_pacific",
        "form4_nwpx",
        "form4_chime_multi_owner",
        "form4a_artesian",
        "form5_asa",
    ],
)
def test_extracts_the_same_xml_the_fixture_holds_separately(fixtures: Path, stem: str) -> None:
    extracted = extract_ownership_xml((fixtures / f"{stem}.txt").read_text())
    assert "<ownershipDocument" in extracted
    assert parse_ownership_document(extracted) == load(fixtures, stem)


def test_a_submission_with_no_ownership_xml_raises(fixtures: Path) -> None:
    """True of filings old enough to predate mandatory XML. Countable, not fatal."""
    submission = (fixtures / "form4_nwpx.txt").read_text()
    stripped = submission.replace("ownershipDocument", "somethingElse")
    with pytest.raises(ParseError):
        extract_ownership_xml(stripped)


def test_does_not_mistake_an_exhibit_for_the_form() -> None:
    submission = (
        "<SEC-DOCUMENT>x.txt : 20260911\n"
        "<DOCUMENT>\n<TYPE>EX-24\n<TEXT>\n<XML>\n"
        "<powerOfAttorney><who>not the form</who></powerOfAttorney>\n"
        "</XML>\n</TEXT>\n</DOCUMENT>\n"
        "<DOCUMENT>\n<TYPE>4\n<TEXT>\n<XML>\n"
        "<ownershipDocument><documentType>4</documentType></ownershipDocument>\n"
        "</XML>\n</TEXT>\n</DOCUMENT>\n"
    )
    assert "ownershipDocument" in extract_ownership_xml(submission)
    assert "powerOfAttorney" not in extract_ownership_xml(submission)


# ------------------------------------------------------------- whole filings


def test_form3_with_true_false_flags(fixtures: Path) -> None:
    filing = load(fixtures, "form3_ambev")
    assert filing.document_type == "3"
    assert filing.is_amendment is False
    assert filing.period_of_report == date(2026, 9, 1)
    assert filing.issuer_cik == 1565025
    assert filing.issuer_name == "AMBEV S.A."
    assert filing.issuer_trading_symbol == "ABEV"

    (owner,) = filing.owners
    assert owner.cik == 2149082
    assert owner.name == "Maffessoni Fernando"
    assert (owner.is_director, owner.is_officer) == (False, True)
    assert owner.officer_title == "See Remarks"
    assert filing.directors == ()


def test_form4_officer_with_entities_in_the_title(fixtures: Path) -> None:
    filing = load(fixtures, "form4_union_pacific")
    (owner,) = filing.owners
    assert owner.cik == 1798280
    assert owner.name == "Hamann Jennifer L"
    assert owner.officer_title == "EVP & CHIEF FINANCIAL OFFICER"
    assert owner.is_officer is True
    assert owner.is_director is False
    assert owner.is_independent is False


def test_omitted_relationship_flags_are_false(fixtures: Path) -> None:
    """NWPX files `isDirector` alone and leaves the other three out entirely."""
    filing = load(fixtures, "form4_nwpx")
    (owner,) = filing.owners
    assert owner.is_director is True
    assert owner.is_officer is False
    assert owner.is_ten_percent_owner is False
    assert owner.is_other is False
    assert owner.officer_title is None
    assert owner.is_independent is True
    assert filing.directors == (owner,)


def test_one_filing_can_have_many_reporting_owners(fixtures: Path) -> None:
    filing = load(fixtures, "form4_chime_multi_owner")
    assert filing.issuer_name == "Chime Financial, Inc."
    assert len(filing.owners) == 10
    assert len({o.cik for o in filing.owners}) == 10
    assert all(o.is_ten_percent_owner for o in filing.owners)
    assert filing.directors == ()


def test_amendment_is_flagged(fixtures: Path) -> None:
    filing = load(fixtures, "form4a_artesian")
    assert filing.document_type == "4/A"
    assert filing.is_amendment is True
    assert filing.owners[0].is_director is True


def test_form5(fixtures: Path) -> None:
    filing = load(fixtures, "form5_asa")
    assert filing.document_type == "5"
    assert filing.period_of_report == date(2025, 11, 30)
    assert filing.issuer_name == "ASA Gold & Precious Metals Ltd"
    assert filing.owners[0].officer_title == "Chief Operating Officer"


def test_ciks_are_integers_not_zero_padded_strings(fixtures: Path) -> None:
    filing = load(fixtures, "form4_union_pacific")
    assert filing.issuer_cik == 100885
    assert isinstance(filing.issuer_cik, int)


def test_a_document_without_an_issuer_cik_raises(fixtures: Path) -> None:
    xml = (fixtures / "form4_nwpx.xml").read_text()
    broken = xml.replace("<issuerCik>0001001385</issuerCik>", "")
    with pytest.raises(ParseError, match="nwpx"):
        parse_ownership_document(broken, accession="nwpx")


# ------------------------------------------------------------------ helpers


def test_element_text_unwraps_footnoteable_values() -> None:
    root = fromstring(
        "<r><a><value>CFO</value><footnoteId id='F1'/></a><b>  CEO  </b><c></c><d>   </d></r>"
    )
    assert element_text(root, "a") == "CFO"
    assert element_text(root, "b") == "CEO"
    assert element_text(root, "c") is None
    assert element_text(root, "d") is None
    assert element_text(root, "missing") is None


def test_element_flag_accepts_both_spellings_and_treats_absence_as_false() -> None:
    root = fromstring(
        "<r><a>1</a><b>0</b><c>true</c><d>false</d><e><value>1</value></e><f> 0 </f></r>"
    )
    assert element_flag(root, "a") is True
    assert element_flag(root, "b") is False
    assert element_flag(root, "c") is True
    assert element_flag(root, "d") is False
    assert element_flag(root, "e") is True
    assert element_flag(root, "f") is False
    assert element_flag(root, "missing") is False


def test_element_flag_refuses_to_guess() -> None:
    """A silently-false isDirector deletes a person from a board."""
    root = fromstring("<r><a>perhaps</a></r>")
    with pytest.raises(ParseError):
        element_flag(root, "a")
