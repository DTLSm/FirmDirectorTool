# EDGAR fixtures

Real documents, fetched on 2026-09-15. Provenance below so they can be
re-fetched or checked. Do not hand-edit: the point of a fixture is that it is
what EDGAR actually served.

## Index files

Both are trimmed — the originals are 737 KB and 55 MB — but every retained line
is byte-for-byte as served, including trailing whitespace, so column offsets
are preserved.

| File | Source |
|---|---|
| `form.20260911.idx` | `Archives/edgar/daily-index/2026/QTR3/form.20260911.idx`, preamble plus 25 selected records |
| `form.2026-QTR2.idx` | `Archives/edgar/full-index/2026/QTR2/form.idx`, first 40 lines |

Between them they cover the awkward parts of the format:

- the daily header wraps across two lines, the quarterly one does not;
- dates are `YYYYMMDD` daily and `YYYY-MM-DD` quarterly;
- `1-A POS` is a form type containing a space;
- accession `0001193125-26-389357` appears on **eleven** lines — one issuer and
  ten reporting owners, all pointing at the same document.

## Ownership documents

Each is kept twice: `.txt` is the complete submission file as EDGAR serves it
(SGML wrapper, headers, the XML inline), `.xml` is just the `ownershipDocument`
extracted from it.

| File | Accession | What it is there for |
|---|---|---|
| `form3_ambev` | 0001193125-26-389339 | Form 3. Boolean flags as `true`/`false`. `officerTitle` is the placeholder `See Remarks` |
| `form4_union_pacific` | 0000100885-26-000268 | Form 4, officer. Flags as `1`/`0`. Title contains an `&amp;` entity |
| `form4_nwpx` | 0001437749-26-030170 | Form 4, director. `isOfficer`, `isTenPercentOwner` and `isOther` are **omitted entirely** |
| `form4_chime_multi_owner` | 0001193125-26-389357 | Form 4 with **ten** reporting owners in one document |
| `form4a_artesian` | 0000863110-26-000091 | Amendment: `documentType` is `4/A` |
| `form5_asa` | 0001213900-26-067096 | Form 5, officer. Carries an empty `<otherText></otherText>` |