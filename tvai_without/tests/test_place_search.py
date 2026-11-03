import io
import json

import pytest

import geocode
import placecheck


def test_parse_coordinates_forms():
    assert geocode.parse_coordinates("27.2046, 77.4977")["lat"] == pytest.approx(27.2046)
    p = geocode.parse_coordinates("12.5 S, 45.1 W")
    assert (p["lat"], p["lon"]) == (-12.5, -45.1)
    d = geocode.parse_coordinates("27°12'16\"N 77°29'51\"E")
    assert d["lat"] == pytest.approx(27.2044, abs=1e-3) and d["lon"] == pytest.approx(77.4975)
    swapped = geocode.parse_coordinates("77°29'51\"E 27°12'16\"N")
    assert swapped["lat"] == pytest.approx(27.2044, abs=1e-3)
    for bad in ["95, 10", "10, 200", "hello", "", "Aravalli Hills"]:
        assert geocode.parse_coordinates(bad) is None


def test_search_place_uses_no_network_for_coordinates(monkeypatch):
    monkeypatch.setattr(geocode.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))
    assert geocode.search_place("10, 20")[0]["kind"] == "coordinates"


class FakeResp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_search_place_parses_nominatim(monkeypatch):
    payload = [{"lat": "27.55", "lon": "76.4", "display_name": "Sariska, Rajasthan, India", "type": "reserve",
                "boundingbox": ["27.3", "27.8", "76.1", "76.7"]}, {"lat": "bad", "lon": "x"}]
    seen = {}

    def fake(req, timeout=0):
        seen["ua"] = req.get_header("User-agent")
        return FakeResp(json.dumps(payload).encode())
    monkeypatch.setattr(geocode.urllib.request, "urlopen", fake)
    out = geocode.search_place("Sariska")
    assert len(out) == 1 and out[0]["bbox"] == (76.1, 27.3, 76.7, 27.8)
    assert "TerraVision" in seen["ua"]


def test_search_place_failure_is_explicit(monkeypatch):
    monkeypatch.setattr(geocode.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(geocode.GeocodeError):
        geocode.search_place("Somewhere")


def fetcher(h, light, errs=()):
    return lambda *a, **k: {"hotspots": h, "light_delta": light, "errors": list(errs)}


HOT = [{"lat": 1, "lon": 1, "temp_k": 330.0, "distance_km": 2.4}]


def test_verdicts():
    assert placecheck.check_place(1, 1, fetcher=fetcher(HOT, 0.0))["status"] == placecheck.FIRE
    assert placecheck.check_place(1, 1, fetcher=fetcher(HOT, 9.0))["status"] == placecheck.FIRE      # fire wins
    assert placecheck.check_place(1, 1, fetcher=fetcher([], 9.0))["status"] == placecheck.LIGHT
    r = placecheck.check_place(1, 1, fetcher=fetcher([], 0.5))
    assert r["status"] == placecheck.CLEAR and r["headline"] == "No anomaly detected"


def test_failures_never_look_like_an_all_clear():
    assert placecheck.check_place(1, 1, fetcher=fetcher(None, None, ["fire check: X"]))["status"] == placecheck.UNKNOWN
    assert placecheck.check_place(1, 1, fetcher=lambda *a, **k: 1 / 0)["status"] == placecheck.UNKNOWN
    partial = placecheck.check_place(1, 1, fetcher=fetcher(None, 0.0, ["fire check: X"]))
    assert partial["status"] == placecheck.CLEAR and "could not run" in partial["detail"]


def test_hotspot_distance():
    assert placecheck.hotspot_distance_km(0, 0, {"lat": 0, "lon": 1}) == pytest.approx(111.19, abs=0.1)
