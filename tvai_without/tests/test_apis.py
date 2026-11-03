import pytest

import apis
import check_apis
import firms
import geocode
import placecheck
import weather

CSV = ("latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
       "27.510,77.200,331.2,0.4,0.4,2026-11-01,742,N,VIIRS,n,2.0NRT,295.1,4.2,D\n"
       "27.520,77.210,300.0,0.4,0.4,2026-11-01,742,N,VIIRS,l,2.0NRT,290.0,1.0,D\n"
       "27.700,77.900,340.0,0.4,0.4,2026-11-01,1105,N,VIIRS,h,2.0NRT,300.0,9.9,D\n")


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ("FIRMS_MAP_KEY", "OPENMETEO_API_KEY", "OPENMETEO_BASE_URL", "GEOCODER_API_KEY", "GEOCODER_URL", "FIRMS_SOURCES"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(apis, "get_setting", lambda n, d="": __import__("os").environ.get(n, d).strip())
    firms._CACHE.clear(); weather._CACHE.clear()


def test_status_reflects_keys(monkeypatch):
    by = {r["id"]: r for r in apis.status_rows()}
    assert by["firms"]["state"] == "missing" and by["weather"]["state"] == "free" and by["cyclones"]["state"] == "free"
    monkeypatch.setenv("FIRMS_MAP_KEY", "abcdef123456")
    monkeypatch.setenv("OPENMETEO_API_KEY", "zzzzzz999999")
    by = {r["id"]: r for r in apis.status_rows()}
    assert by["firms"]["state"] == "live" and by["weather"]["state"] == "live"


def test_keys_are_redacted(monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "SECRETKEY123")
    assert "SECRETKEY123" not in apis.redact("failed for https://x/api/SECRETKEY123/VIIRS")
    assert "***" in apis.redact("token SECRETKEY123")


def test_firms_parse_filter_and_dedupe():
    hs = firms.parse_csv(CSV, "VIIRS_SNPP_NRT")
    assert len(hs) == 3 and hs[0]["time"] == "0742" and hs[0]["temp_k"] == pytest.approx(331.2)
    got = firms.fetch_hotspots((77, 27, 78, 28), 3, sources=["A", "B"], key="K" * 10, http=lambda u: CSV)
    assert len(got) == 2                                  # low confidence dropped; the second source's copies de-duplicated
    kept = firms.fetch_hotspots((76, 26, 79, 29), 3, sources=["A"], key="K" * 10, http=lambda u: CSV, drop_low_confidence=False)
    assert len(kept) == 3


def test_firms_url_days_and_errors(monkeypatch):
    seen = []
    firms.fetch_hotspots((77, 27, 78, 28), 99, sources=["VIIRS_SNPP_NRT"], key="MYKEY12345", http=lambda u: seen.append(u) or CSV)
    assert seen[0].endswith("/MYKEY12345/VIIRS_SNPP_NRT/77.0000,27.0000,78.0000,28.0000/5")      # days clamped to 5
    with pytest.raises(firms.FirmsError):
        firms.fetch_hotspots((0, 0, 1, 1), 1, key=None)                                       # no key configured
    monkeypatch.setattr(firms, "call_with_retry", lambda fn, *a, **k: fn(*a))
    with pytest.raises(firms.FirmsError) as e:
        firms.fetch_hotspots((0, 0, 1, 1), 1, sources=["A"], key="BADKEY99999", http=lambda u: "Invalid MAP_KEY.")
    assert "BADKEY99999" not in str(e.value)


def test_confidence_rules():
    assert firms.is_low_confidence({"confidence": "l"}) and firms.is_low_confidence({"confidence": "20"})
    assert not firms.is_low_confidence({"confidence": "h"}) and not firms.is_low_confidence({"confidence": "80"})


def test_place_check_uses_firms_when_key_set(monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "KEYKEYKEY1")
    monkeypatch.setattr(firms, "_http_text", lambda url, timeout=40: CSV)
    monkeypatch.setattr(placecheck, "_light_from_ee", lambda *a, **k: 0.5)
    monkeypatch.setattr(placecheck, "_fire_from_ee", lambda *a, **k: (_ for _ in ()).throw(AssertionError("EE fire used")))
    r = placecheck.check_place(27.51, 77.2, radius_km=10, days=2)
    assert r["status"] == placecheck.FIRE and r["fire_source"] == "NASA FIRMS API"
    assert r["hotspots"][0]["when"] == "2026-11-01 07:42 UTC" and len(r["hotspots"]) == 1     # the 77.9° one is > 10 km away


def test_firms_failure_falls_back_and_never_claims_clear(monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "KEYKEYKEY1")
    monkeypatch.setattr(firms, "call_with_retry", lambda fn, *a, **k: fn(*a))
    monkeypatch.setattr(firms, "_http_text", lambda url, timeout=40: "Invalid MAP_KEY.")
    monkeypatch.setattr(placecheck, "_fire_from_ee", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ee")))
    monkeypatch.setattr(placecheck, "_light_from_ee", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ee")))
    r = placecheck.check_place(27.5, 77.2)
    assert r["status"] == placecheck.UNKNOWN and "KEYKEYKEY1" not in " ".join(r["errors"])


def test_weather_paid_key_switches_host(monkeypatch):
    urls = []
    def http(url):
        urls.append(url)
        return {"hourly": {"time": ["2026-11-02T00:00"], "temperature_2m": [30]}}
    weather.fetch_forecast([{"lat": 1.0, "lon": 2.0}], 24, http)
    assert urls[0].startswith("https://api.open-meteo.com/") and "apikey" not in urls[0]
    monkeypatch.setenv("OPENMETEO_API_KEY", "PAIDKEY123456")
    weather._CACHE.clear()
    weather.fetch_forecast([{"lat": 1.0, "lon": 2.0}], 24, http)
    assert urls[1].startswith("https://customer-api.open-meteo.com/") and "apikey=PAIDKEY123456" in urls[1]


def test_geocoder_key_and_custom_url(monkeypatch):
    seen = {}
    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'[{"lat":"1.5","lon":"2.5","display_name":"X","boundingbox":["1","2","3","4"],"type":"city"}]'
    monkeypatch.setattr(geocode.urllib.request, "urlopen", lambda req, timeout=0: seen.update(url=req.full_url) or R())
    monkeypatch.setenv("GEOCODER_URL", "https://us1.locationiq.com/v1/search")
    monkeypatch.setenv("GEOCODER_API_KEY", "GEOKEY123456")
    out = geocode.search_place("Somewhere")
    assert seen["url"].startswith("https://us1.locationiq.com/v1/search?") and "key=GEOKEY123456" in seen["url"]
    assert out[0]["bbox"] == (3.0, 1.0, 4.0, 2.0)


def test_check_apis_reports_without_crashing(monkeypatch):
    def boom(_=None): raise RuntimeError("down")
    res = {r["id"]: r for r in check_apis.run({"weather": lambda _=None: "ok", "geocoder": boom, "cyclones": lambda _=None: "ok", "gee": boom})}
    assert res["weather"]["ok"] is True and res["geocoder"]["ok"] is False
    assert res["firms"]["ok"] is None                      # no key: skipped, not failed


def test_geocoder_key_never_goes_to_the_default_host(monkeypatch):
    monkeypatch.setenv("GEOCODER_API_KEY", "GEOKEY123456")                 # key set, but no URL
    assert apis.geocoder_endpoint()["key"] == "" and "openstreetmap" in apis.geocoder_endpoint()["url"]
    row = {r["id"]: r for r in apis.status_rows()}["geocoder"]
    assert row["state"] == "missing" and "GEOCODER_URL" in row["detail"]
    monkeypatch.setenv("GEOCODER_URL", "https://us1.locationiq.com/v1/search")
    assert apis.geocoder_endpoint()["key"] == "GEOKEY123456"
