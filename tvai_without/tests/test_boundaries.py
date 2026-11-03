import json

import boundaries
from evidence import score_evidence, zone_of

SQUARE_WITH_HOLE = {
    "type": "Polygon",
    "coordinates": [
        [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
        [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]],
    ],
}


def bnd(geom=SQUARE_WITH_HOLE, source="wdpa"):
    return {"source": source, "total": 1, "truncated": False, "fetched_at": "",
            "features": [{"type": "Feature", "properties": {"NAME": "Test Reserve"}, "geometry": geom}]}


def test_point_in_polygon_respects_holes():
    inside, names = boundaries.points_inside(bnd(), [1, 5, 20], [1, 5, 5])
    assert list(inside) == [True, False, False]          # centre is a hole, third is outside
    assert names[0] == "Test Reserve" and names[1] == ""


def test_multipolygon():
    mp = {"type": "MultiPolygon", "coordinates": [
        [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        [[[5, 5], [6, 5], [6, 6], [5, 6], [5, 5]]]]}
    inside, _ = boundaries.points_inside(bnd(mp), [0.5, 5.5, 3], [0.5, 5.5, 3])
    assert list(inside) == [True, True, False]


def test_annotate_official_and_wdpa_fallback():
    a = [{"lat": 1, "lon": 1}, {"lat": 20, "lon": 20}]
    boundaries.annotate_alerts(a, bnd(source="official"))
    assert a[0]["in_official_boundary"] is True and a[1]["in_official_boundary"] is False
    b = [{"lat": 1, "lon": 1}, {"lat": 20, "lon": 20, "in_protected_area": True}]
    boundaries.annotate_alerts(b, bnd(), fill_protected_area=True)
    assert b[0]["in_protected_area"] is True and b[1]["in_protected_area"] is True   # existing value kept
    c = [{"lat": 1, "lon": 1}]
    boundaries.annotate_alerts(c, bnd())
    assert "in_protected_area" not in c[0]


def test_official_file_overrides_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(boundaries, "OFFICIAL_DIR", tmp_path / "off")
    monkeypatch.setattr(boundaries, "CACHE_DIR", tmp_path / "cache")
    region = "Aravalli Hills, India"
    boundaries.CACHE_DIR.mkdir(parents=True)
    boundaries.cache_path(region).write_text(json.dumps(
        {"fetched_at": "2026-01-01 00:00", "total": 1, "features": bnd()["features"]}))
    assert boundaries.load_boundary(region)["source"] == "wdpa"
    boundaries.OFFICIAL_DIR.mkdir(parents=True)
    boundaries.official_path(region).write_text(json.dumps(bnd()["features"][0]))
    assert boundaries.load_boundary(region)["source"] == "official"
    boundaries.official_path(region).write_text("not json")      # broken file falls back, never crashes
    assert boundaries.load_boundary(region)["source"] == "wdpa"


def test_ensure_boundary_failure_is_graceful(tmp_path, monkeypatch):
    monkeypatch.setattr(boundaries, "OFFICIAL_DIR", tmp_path / "off")
    monkeypatch.setattr(boundaries, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(boundaries, "fetch_wdpa_boundary", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ee")))
    b, note = boundaries.ensure_boundary("Aravalli Hills, India")
    assert b is None and "search box" in note


def test_zone_logic():
    assert zone_of({}) is None
    assert zone_of({"in_official_boundary": True, "in_protected_area": False}) == "official"
    assert zone_of({"in_official_boundary": False, "in_protected_area": True}) == "outside"
    assert zone_of({"in_protected_area": True}) == "protected"
    assert zone_of({"in_protected_area": False, "tree_context": 0.6}) == "forest"
    assert zone_of({"in_protected_area": False, "tree_context": 0.05}) == "outside"
    assert zone_of({"tree_context": 0.8}) == "forest"


def test_forest_zone_is_not_penalised():
    base = {"category": "illegal_encroachment", "model_confidence": 70}
    out = score_evidence({**base, "in_protected_area": False, "tree_context": 0.7})
    assert out["confidence"] == 70 and out["zone"] == "forest"
    assert score_evidence({**base, "in_protected_area": False, "tree_context": 0.0})["confidence"] == 60
