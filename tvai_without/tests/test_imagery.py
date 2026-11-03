from datetime import date

import imagery


def test_windows_cover_the_same_season_one_year_apart():
    w = imagery.image_windows(date(2026, 10, 17), 120)
    assert w["after"] == (date(2026, 6, 19), date(2026, 10, 17))
    assert w["before"] == (date(2025, 6, 19), date(2025, 10, 17))
    for key in w:
        assert (w[key][1] - w[key][0]).days == 120


def test_leap_day_dates_do_not_crash():
    w = imagery.image_windows(date(2028, 2, 29), 90)
    assert w["before"][1] < w["after"][1]
