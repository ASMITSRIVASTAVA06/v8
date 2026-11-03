import json
from datetime import datetime

import pytest

import cyclones
import riskzones
import scheduled_weather
import weather
import weather_alerts

T = [f"2026-11-02T{h % 24:02d}:00" for h in range(72)]


def hourly(**over):
    base = {"time": T, "temperature_2m": [30.0] * 72, "precipitation": [0.0] * 72, "wind_gusts_10m": [15.0] * 72,
            "cape": [100.0] * 72, "weather_code": [1] * 72}
    base.update(over)
    return base


# ── scoring ──────────────────────────────────────────────────────────────────

def test_calm_day_is_low():
    r = weather.assess_point(hourly())
    assert r["level"] == 0 and "No significant" in r["headline"]


def test_rain_classes_follow_imd_thresholds():
    def lvl(total):                       # spread over 24 hours so the hourly-intensity rule does not trigger
        return weather.assess_point(hourly(precipitation=[total / 24] * 24 + [0] * 48))["hazards"]["rain"]["level"]
    assert [lvl(10), lvl(70), lvl(120), lvl(210)] == [0, 1, 2, 3]
    assert weather.assess_point(hourly(precipitation=[35.0] + [0] * 71))["hazards"]["rain"]["level"] == 2   # cloudburst-style hour


def test_thunderstorm_needs_a_storm_not_just_energy():
    energy_only = weather.assess_point(hourly(cape=[3000.0] * 72))
    assert energy_only["hazards"]["storm"]["level"] == 1             # a Watch at most
    storm = weather.assess_point(hourly(cape=[1500.0] * 72, weather_code=[1] * 10 + [95] * 2 + [1] * 60))
    assert storm["hazards"]["storm"]["level"] == 2
    hail = weather.assess_point(hourly(cape=[1500.0] * 72, weather_code=[96] * 2 + [1] * 70))
    assert hail["hazards"]["storm"]["level"] == 3


def test_wind_heat_and_horizon():
    r = weather.assess_point(hourly(wind_gusts_10m=[15.0] * 70 + [125.0, 20.0], temperature_2m=[44.0] * 72))
    assert r["hazards"]["wind"]["level"] == 3 and r["hazards"]["heat"]["level"] == 2
    short = weather.assess_point(hourly(wind_gusts_10m=[15.0] * 70 + [125.0, 20.0]), hours=24)
    assert short["hazards"]["wind"]["level"] == 0                    # the gust is beyond the 24 h window


def test_missing_values_do_not_crash():
    r = weather.assess_point({"time": T[:3], "cape": [None, "x", float("nan")], "precipitation": None})
    assert r["level"] == 0


# ── grid and fetching ────────────────────────────────────────────────────────

def test_grid_is_capped_and_covers_bounds():
    g = weather.make_grid((-75, -15, -45, 5), 0.25, max_points=100)
    assert 0 < len(g) <= 100
    assert min(c["lat"] - c["half_lat"] for c in g) == pytest.approx(-15, abs=1e-3) and max(c["lon"] + c["half_lon"] for c in g) == pytest.approx(-45, abs=1e-3)


def fake_http(calls):
    def http(url):
        calls.append(url)
        n = url.split("latitude=")[1].split("&")[0].count(",") + 1
        items = [{"hourly": hourly()} for _ in range(n)]
        return items if n > 1 else items[0]
    return http


def test_fetch_batches_caches_and_aligns():
    weather._CACHE.clear()
    pts = [{"lat": i * 0.5, "lon": 10.0} for i in range(45)]
    calls: list = []
    out, errs = weather.fetch_forecast(pts, 72, fake_http(calls), now=1000.0)
    assert len(calls) == 3 and all(o is not None for o in out) and errs == []     # 20 + 20 + 5
    weather.fetch_forecast(pts, 72, fake_http(calls), now=1100.0)
    assert len(calls) == 3                                                       # served from the cache
    weather.fetch_forecast(pts[:1], 72, fake_http(calls), now=1000.0 + weather.CACHE_TTL_S + 5)
    assert len(calls) == 4                                                       # expired


def test_fetch_failure_marks_points_unknown_not_safe(monkeypatch):
    weather._CACHE.clear()
    monkeypatch.setattr(weather, "call_with_retry", lambda fn, *a, **k: fn(*a))
    def broken(url):
        raise OSError("offline")
    out, errs = weather.fetch_forecast([{"lat": 1.0, "lon": 1.0}], 72, broken)
    assert out == [None] and errs
    area = weather.assess_area((0, 0, 2, 2), 1.0, 72, broken)
    assert area["cells"] == [] and area["missing"] > 0


def test_shape_mismatch_is_rejected(monkeypatch):
    weather._CACHE.clear()
    monkeypatch.setattr(weather, "call_with_retry", lambda fn, *a, **k: fn(*a))
    out, _ = weather.fetch_forecast([{"lat": 3.0, "lon": 3.0}], 72, lambda url: {"error": True, "reason": "limit"})
    assert out == [None]


# ── cyclones ─────────────────────────────────────────────────────────────────

EVENTS = {"features": [
    {"properties": {"eventid": 1001, "eventname": "TEST", "alertlevel": "Orange", "iscurrent": "true",
                    "severitydata": {"severity": 120, "severitytext": "Tropical storm"}, "source": "NOAA"},
     "geometry": {"coordinates": [88.0, 15.0]}},
    {"properties": {"eventid": 1002, "eventname": "OLD", "alertlevel": "Green", "iscurrent": "false"},
     "geometry": {"coordinates": [60.0, 10.0]}},
    {"properties": {"eventid": 1003}, "geometry": {}},
]}
TIMELINE = {"channel": {"item": [
    {"latitude": "15", "longitude": "88", "actual": "True", "wind_speed": "50", "windrad_nm_34kt_ne": "90"},
    {"latitude": "19", "longitude": "86", "actual": "False", "wind_speed": "70", "windrad_nm_34kt_ne": "120",
     "windrad_nm_34kt_se": "100", "storm_status": ["Hurricane"]}]}}


def cyc_http(url):
    if "geteventlist" in url:
        return EVENTS
    if "geteventdata" in url:
        return {"properties": {"impacts": [{"resource": {"timeline": "https://example.test/timeline.json"}}]}}
    return TIMELINE


def test_parse_events_and_active_filter():
    storms, notes = cyclones.list_active_cyclones(cyc_http)
    assert [s["name"] for s in storms] == ["TEST"]                    # not-current and malformed events dropped
    assert len(storms[0]["track"]) == 2 and storms[0]["track"][1]["radius_km"] == pytest.approx(222.2)


def test_feed_down_raises_but_missing_track_degrades(monkeypatch):
    monkeypatch.setattr(cyclones, "call_with_retry", lambda fn, *a, **k: fn(*a))
    with pytest.raises(cyclones.CycloneError):
        cyclones.list_active_cyclones(lambda u: (_ for _ in ()).throw(OSError("offline")))
    def no_track(url):
        if "geteventdata" in url:
            raise OSError("track failed")
        return cyc_http(url)
    storms, notes = cyclones.list_active_cyclones(no_track)
    assert storms and len(storms[0]["track"]) == 1 and any("track unavailable" in n for n in notes)


def test_cyclone_levels():
    s = cyclones.list_active_cyclones(cyc_http)[0][0]
    assert cyclones.assess_cyclone(20, 86, s)["level"] == 3           # inside the hurricane's wind field
    assert cyclones.assess_cyclone(23, 86, s)["level"] == 1           # 444 km from the track: within 600 km
    assert cyclones.assess_cyclone(26, 86, s)["level"] == 0           # 778 km away
    assert cyclones.assess_cyclone(26, 77, s)["level"] == 0
    assert cyclones.worst_cyclone_threat(1, 1, []) is None


# ── zones, alerts and the scheduled job ──────────────────────────────────────

def test_combine_raises_cell_level_for_cyclone():
    s = cyclones.list_active_cyclones(cyc_http)[0]
    cells = [{"lat": 20.0, "lon": 86.0, "level": 0, "hazards": weather.assess_point(hourly())["hazards"], "headline": ""}]
    riskzones.combine(cells, s)
    assert cells[0]["level"] == 3 and "cyclone" in cells[0]["hazards"] and "cyclone" in cells[0]["headline"].lower()


def test_region_overview_isolates_failures(monkeypatch):
    def fake_assess(bounds, storms, step, hours, http):
        if bounds == riskzones.REGION_BOUNDS["Amazon Basin, Brazil"]:
            raise RuntimeError("boom")
        return {"cells": [{"lat": 0, "lon": 0, "level": 2, "hazards": weather.assess_point(hourly(wind_gusts_10m=[95.0] * 72))["hazards"],
                           "headline": "Warning: damaging wind"}], "errors": [], "worst": 2}
    monkeypatch.setattr(riskzones, "assess_bounds", fake_assess)
    rows = riskzones.region_overview(regions=["Amazon Basin, Brazil", "Black Forest, Germany"])
    by = {r["region"]: r for r in rows}
    assert by["Amazon Basin, Brazil"]["level"] is None and by["Black Forest, Germany"]["level"] == 2
    assert rows[0]["region"] == "Black Forest, Germany"                # failures sort last


ROWS = [{"region": "Western Ghats, India", "level": 3, "headline": "Severe: cyclone", "flagged": 4, "cells": 9, "peak_at": "2026-11-03T06:00"},
        {"region": "Black Forest, Germany", "level": 0, "headline": "calm", "flagged": 0, "cells": 4, "peak_at": ""}]
STORM = [{"name": "TEST", "alert": "Orange", "severity_text": "Tropical storm"}]


def test_message_only_when_needed_and_not_repeated(tmp_path):
    assert weather_alerts.build_message([ROWS[1]], [], 2) is None
    m = weather_alerts.build_message(ROWS, STORM, 2)
    assert "Western Ghats" in m["body"] and "Black Forest" not in m["body"] and "official warning" in m["body"]
    state = {}
    assert weather_alerts.should_send(m, state)
    now = datetime(2026, 11, 2, 12, 0)
    path = tmp_path / "s.json"
    weather_alerts.save_state(m, path, now)
    state = weather_alerts.load_state(path)
    assert not weather_alerts.should_send(m, state, now=datetime(2026, 11, 2, 18, 0))
    assert weather_alerts.should_send(m, state, now=datetime(2026, 11, 3, 13, 0))        # Severe re-sent after 24 h
    other = weather_alerts.build_message([{**ROWS[0], "region": "Borneo Rainforest"}], STORM, 2)
    assert weather_alerts.should_send(other, state)


def test_scheduled_job_dry_run(monkeypatch, capsys):
    storms = cyclones.list_active_cyclones(cyc_http)[0]
    monkeypatch.setattr(cyclones, "list_active_cyclones", lambda: (storms, []))
    monkeypatch.setattr(riskzones, "region_overview", lambda storms, hours, regions: ROWS)
    monkeypatch.setattr(scheduled_weather, "parse_regions", lambda t: ["Western Ghats, India", "Black Forest, Germany"])
    assert scheduled_weather.main(["--regions", "all", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[dry run]" in out and "Western Ghats" in out


def test_scheduled_job_reports_total_failure(monkeypatch):
    monkeypatch.setattr(cyclones, "list_active_cyclones", lambda: (_ for _ in ()).throw(cyclones.CycloneError("down")))
    monkeypatch.setattr(riskzones, "region_overview", lambda storms, hours, regions: [{"region": "X", "level": None, "headline": "", "flagged": 0, "cells": 0}])
    monkeypatch.setattr(scheduled_weather, "parse_regions", lambda t: ["Amazon Basin, Brazil"])
    assert scheduled_weather.main(["--dry-run"]) == 1
