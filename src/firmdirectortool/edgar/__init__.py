"""SEC EDGAR ingestion: HTTP client, index walkers, and Section 16 parsers."""

from .cache import RawCache
from .client import (
    SEC_RATE_LIMIT,
    ClientConfig,
    EdgarClient,
    backoff_delay,
    retry_after_seconds,
)
from .errors import (
    BlockedError,
    EdgarConnectionError,
    EdgarError,
    EdgarHTTPError,
    NotFoundError,
    ParseError,
    RateLimitError,
)
from .indexes import (
    OWNERSHIP_FORMS,
    IndexEntry,
    daily_index_url,
    parse_form_idx,
    quarter_of,
    quarterly_index_url,
    walk_daily,
    walk_quarterly,
)
from .parse import (
    OwnershipFiling,
    ReportingOwner,
    extract_ownership_xml,
    parse_ownership_document,
)
from .ratelimit import TokenBucket
from .submissions import EntitySubmissions, fetch_submissions, parse_submissions, submissions_url

__all__ = [
    "OWNERSHIP_FORMS",
    "SEC_RATE_LIMIT",
    "BlockedError",
    "ClientConfig",
    "EdgarClient",
    "EdgarConnectionError",
    "EdgarError",
    "EdgarHTTPError",
    "EntitySubmissions",
    "IndexEntry",
    "NotFoundError",
    "OwnershipFiling",
    "ParseError",
    "RateLimitError",
    "RawCache",
    "ReportingOwner",
    "TokenBucket",
    "backoff_delay",
    "daily_index_url",
    "extract_ownership_xml",
    "fetch_submissions",
    "parse_form_idx",
    "parse_ownership_document",
    "parse_submissions",
    "quarter_of",
    "quarterly_index_url",
    "retry_after_seconds",
    "submissions_url",
    "walk_daily",
    "walk_quarterly",
]
