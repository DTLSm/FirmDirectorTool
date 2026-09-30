from collections.abc import Sequence
from datetime import date

import pytest

from firmdirectortool.ingest.__main__ import default_window, parse_args

TODAY = date(2026, 9, 30)


def window(argv: Sequence[str]) -> tuple[date, date]:
    """The start and end dates ``argv`` asks for."""
    options = parse_args(argv, TODAY)
    return options.start, options.end


def test_the_default_window_is_the_fourteen_days_before_today():
    assert default_window(date(2026, 9, 30)) == (date(2026, 9, 16), date(2026, 9, 29))


def test_the_default_window_crosses_a_month_boundary():
    assert default_window(date(2026, 3, 5)) == (date(2026, 2, 19), date(2026, 3, 4))


def test_no_dates_means_the_default_window():
    assert window([]) == default_window(TODAY)


def test_both_dates_are_taken_as_given():
    args = ["--start", "2026-09-01", "--end", "2026-09-05"]
    assert window(args) == (date(2026, 9, 1), date(2026, 9, 5))


def test_start_alone_runs_up_to_yesterday():
    assert window(["--start", "2026-09-01"]) == (date(2026, 9, 1), date(2026, 9, 29))


def test_start_after_end_is_refused():
    with pytest.raises(SystemExit):
        parse_args(["--start", "2026-09-05", "--end", "2026-09-01"], TODAY)


def test_a_date_that_does_not_exist_is_refused():
    with pytest.raises(SystemExit):
        parse_args(["--start", "2026-13-01"], TODAY)


def test_a_normal_run_is_not_a_retry():
    assert parse_args([], TODAY).retry_errors is False


def test_retry_errors_is_a_switch():
    assert parse_args(["--retry-errors"], TODAY).retry_errors is True


def test_retry_errors_with_dates_is_refused():
    with pytest.raises(SystemExit):
        parse_args(["--retry-errors", "--start", "2026-09-01"], TODAY)
