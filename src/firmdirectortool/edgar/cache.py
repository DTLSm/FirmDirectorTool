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
one exception — a daily index for today is still growing — which is why
:meth:`RawCache.put` is a separate call the client can skip.
"""

from __future__ import annotations

from pathlib import Path


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
        """Map a URL to the file that holds (or would hold) its body.

        Implementation notes:

        * Parse with :func:`urllib.parse.urlsplit`; use ``netloc`` and ``path``
          and ignore the fragment.
        * A query string must not be dropped — ``.../index.json?foo`` and
          ``.../index.json`` are different resources. Append it in a form that
          is legal in a filename (percent-encoding the separator is enough).
        * Refuse to escape ``root``. A path containing ``..`` segments, or an
          absolute-looking ``netloc``, must not produce a path outside
          ``self.root``; resolve and check, and raise :class:`ValueError` if it
          would. EDGAR will not send such a URL, but the cache root is going to
          be a mounted volume and later a storage account, and a cache that can
          be talked into writing anywhere is a real vulnerability.
        * A URL ending in ``/`` has no filename. Pick a deterministic one
          (``index.html`` is conventional) rather than creating a directory and
          failing later.
        """
        raise NotImplementedError

    def get(self, url: str) -> bytes | None:
        """Return the cached body, or ``None`` if it has not been fetched.

        Must not raise when the cache is cold or the root does not exist yet —
        a miss is the normal case on a fresh checkout.
        """
        raise NotImplementedError

    def put(self, url: str, body: bytes) -> Path:
        """Write ``body`` and return the path written.

        Create parent directories as needed. Write atomically: a crash halfway
        through a multi-megabyte submission file must not leave a truncated
        document that a later run happily treats as a cache hit. Write to a
        temporary file in the same directory and :func:`os.replace` it into
        place — ``os.replace`` is atomic within a filesystem, ``shutil.move``
        across filesystems is not.
        """
        raise NotImplementedError
