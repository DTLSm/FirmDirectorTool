"""Exception hierarchy for the EDGAR client.

The distinction that matters operationally is *retryable* versus *terminal*.
A 429 or a 5xx is a transient condition: back off and try again. A 404 means
the document is not there and never will be. A 403 from ``sec.gov`` almost
always means the ``User-Agent`` was missing or malformed, or that a previous
run exceeded the rate limit badly enough to earn a block — retrying makes it
worse, so it is terminal and says so loudly.
"""

from __future__ import annotations


class EdgarError(Exception):
    """Base class for everything this package raises."""


class EdgarHTTPError(EdgarError):
    """An HTTP response the client could not turn into a document."""

    def __init__(self, status_code: int, url: str, message: str | None = None) -> None:
        self.status_code = status_code
        self.url = url
        super().__init__(message or f"HTTP {status_code} for {url}")


class NotFoundError(EdgarHTTPError):
    """404. Expected for daily index files on weekends and market holidays."""


class RateLimitError(EdgarHTTPError):
    """429 that the client stopped retrying.

    Either the retry budget was spent, or the server's ``Retry-After`` asked
    for a longer wait than ``ClientConfig.max_retry_after`` allows.
    """


class BlockedError(EdgarHTTPError):
    """403. Fix the User-Agent or slow down; do not retry."""

    def __init__(self, url: str, message: str | None = None) -> None:
        super().__init__(
            403,
            url,
            message
            or (
                f"403 from {url}. The SEC blocks requests without a descriptive "
                "User-Agent, and blocks clients that exceed the fair-access limit. "
                "Check EDGAR_USER_AGENT and the rate limiter before retrying."
            ),
        )


class EdgarConnectionError(EdgarError):
    """No response at all, still, after the retry budget was spent.

    Refused, reset, timed out, unresolvable. Deliberately not an
    :class:`EdgarHTTPError`: there is no status code to report, and inventing
    one would mislead whoever reads the log. The underlying ``httpx`` exception
    is chained as ``__cause__``.
    """

    def __init__(self, url: str, message: str | None = None) -> None:
        self.url = url
        super().__init__(message or f"no response from {url}")


class ParseError(EdgarError):
    """A document was fetched but could not be understood."""
