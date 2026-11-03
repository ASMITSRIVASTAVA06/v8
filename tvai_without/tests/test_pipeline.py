from datetime import date

import pipeline


def test_rolling_window_two_months():
    start, end, label = pipeline.rolling_window(date(2026, 8, 1))
    assert (start, end, label) == (date(2026, 7, 1), date(2026, 9, 1), "2026-08")


def test_rolling_window_crosses_year_boundary():
    start, end, label = pipeline.rolling_window(date(2026, 1, 1))
    assert (start, end, label) == (date(2025, 12, 1), date(2026, 2, 1), "2026-01")
    assert pipeline.rolling_window(date(2026, 12, 15))[1] == date(2027, 1, 1)
