"""On-disk cache of raw EDGAR responses.

Every document is written to disk exactly as it came off the wire, keyed by
URL, *before* anything parses it. Two reasons:

1. **Re-parsing is free; re-fetching is not.** Parser bugs are discovered late
   and fixed often. Each fix would otherwise mean walking the same index pages
   and the same few hundred thousand filings again, at 10 requests per second,
   for no new information.
2. **Reproducibility.** A distance computed today can be traced back to the
   exact bytes it came from, which is the difference between a research tool
   and a black box.

The layout mirrors the URL — ``<root>/<host>/<path>`` — rather than hashing it.
A hash would be shorter and collision-free, but being able to ``ls`` the cache
and see ``www.sec.gov/Archives/edgar/data/320193/...`` is worth more during
development than the saved bytes. It also maps one-to-one onto Blob Storage
prefixes in Slice 2, so the local cache and the cloud object store use the same
keys.

The cache is deliberately *not* invalidated. EDGAR accessions are immutable
once filed; a changed filing gets a new accession number. Index files are the
one exception — a daily index for today is still growing — which is why the
client can be told to distrust a cached copy on a per-request basis.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import quote, urlsplit

#: Stands in for the filename when a URL names a directory rather than a file.
DIRECTORY_INDEX = "index.html"

#: Percent-encoded "?", so a query string becomes part of the filename instead
#: of a path separator.
QUERY_MARKER = "%3F"


class RawCache:
    """A filesystem cache keyed by URL.

    Parameters
    ----------
    root
        Directory under which everything is written. Created on demand.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, url: str) -> Path:
        """Map a URL to the file that holds (or would hold) its body."""
        parts = urlsplit(url)
        if not parts.netloc:
            raise ValueError(f"cannot cache a URL with no host: {url!r}")

        path = parts.path
        if not path or path.endswith("/"):
            # A directory URL has no filename of its own. Inventing one
            # deterministically beats creating a directory here and colliding
            # with a real file later.
            path += DIRECTORY_INDEX
        if parts.query:
            # ".../submissions/CIK0000320193.json" and the same URL with "?v=2"
            # are different resources and must not share a file. The fragment
            # is dropped: servers never see it.
            path += QUERY_MARKER + quote(parts.query, safe="")

        candidate = self.root / parts.netloc / path.lstrip("/")
        normalised = Path(os.path.normpath(candidate))

        # A URL path may contain ".." segments. EDGAR will not send one, but the
        # cache root is a mounted volume today and a storage account tomorrow,
        # and a cache that can be talked into writing outside its root is a real
        # vulnerability. normpath collapses the traversal; resolve() then
        # follows any symlinks before the containment check, so neither trick
        # gets through.
        if not normalised.resolve().is_relative_to(self.root.resolve()):
            raise ValueError(f"{url!r} maps outside the cache root {self.root}")
        return normalised

    def get(self, url: str) -> bytes | None:
        """Return the cached body, or ``None`` if it has not been fetched.

        A cold cache is the normal case on a fresh checkout, so a miss is a
        return value rather than an exception. A URL that maps outside the root
        still raises: that is a bug, not a miss.
        """
        try:
            return self.path_for(url).read_bytes()
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            return None

    def put(self, url: str, body: bytes) -> Path:
        """Write ``body`` and return the path written.

        The write is atomic. A crash halfway through a multi-megabyte
        submission file must not leave a truncated document that a later run
        treats as a cache hit — that failure is silent, survives restarts, and
        looks like a parser bug. Writing to a temporary file in the *same*
        directory and calling :func:`os.replace` avoids it: ``os.replace`` is
        atomic within a filesystem, where :func:`shutil.move` across
        filesystems is not.
        """
        path = self.path_for(url)
        path.parent.mkdir(parents=True, exist_ok=True)

        handle, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp", suffix=".part")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle, "wb") as sink:
                sink.write(body)
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return path
