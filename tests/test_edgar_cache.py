from pathlib import Path

import pytest

from firmdirectortool.edgar import RawCache

URL = "https://www.sec.gov/Archives/edgar/data/320193/0000320193-26-000008.txt"


def test_path_mirrors_the_url_under_the_root(tmp_path: Path) -> None:
    cache = RawCache(tmp_path)
    expected = tmp_path / "www.sec.gov/Archives/edgar/data/320193/0000320193-26-000008.txt"
    assert cache.path_for(URL) == expected


def test_query_string_distinguishes_two_resources(tmp_path: Path) -> None:
    cache = RawCache(tmp_path)
    plain = cache.path_for("https://data.sec.gov/submissions/CIK0000320193.json")
    queried = cache.path_for("https://data.sec.gov/submissions/CIK0000320193.json?v=2")
    assert plain != queried
    assert queried.parent == plain.parent


def test_directory_url_gets_a_filename(tmp_path: Path) -> None:
    cache = RawCache(tmp_path)
    path = cache.path_for("https://www.sec.gov/Archives/edgar/data/320193/")
    assert path.name
    assert not str(path).endswith("/")


def test_refuses_to_escape_the_root(tmp_path: Path) -> None:
    cache = RawCache(tmp_path / "cache")
    with pytest.raises(ValueError):
        cache.path_for("https://www.sec.gov/../../../etc/passwd")


def test_miss_on_a_cold_cache_returns_none_and_does_not_raise(tmp_path: Path) -> None:
    cache = RawCache(tmp_path / "does-not-exist-yet")
    assert cache.get(URL) is None


def test_put_then_get_round_trips_exact_bytes(tmp_path: Path) -> None:
    cache = RawCache(tmp_path)
    body = b"<SEC-DOCUMENT>\xff\xfe binary-ish \r\n not utf-8"
    written = cache.put(URL, body)
    assert written.read_bytes() == body
    assert cache.get(URL) == body


def test_put_leaves_no_temporary_files_behind(tmp_path: Path) -> None:
    """A partial write must never be mistaken for a cache hit by a later run."""
    cache = RawCache(tmp_path)
    written = cache.put(URL, b"body")
    assert sorted(p.name for p in written.parent.iterdir()) == [written.name]
