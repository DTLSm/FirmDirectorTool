from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures() -> Path:
    """Directory of real EDGAR documents. See its README for provenance."""
    return Path(__file__).parent / "fixtures" / "edgar"
