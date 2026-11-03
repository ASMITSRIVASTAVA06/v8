"""Exercises add_context's chunking and failure isolation with a stand-in for Earth Engine."""
import sys
import types

import enrichment


def fake_ee():
    m = types.ModuleType("ee")
    m.Geometry = types.SimpleNamespace(Point=lambda c: c)
    m.Feature = lambda g, p=None: {"g": g, "p": p}
    m.FeatureCollection = lambda feats: feats
    return m


def test_failed_layer_is_reported_and_others_still_apply(monkeypatch=None):
    sys.modules["ee"] = fake_ee()
    saved = enrichment.LAYER_FUNCS
    try:
        def ok(ee, pts, today):
            return {f["p"]["i"]: {"builtup_fraction": 0.1} for f in pts}

        def boom(ee, pts, today):
            raise RuntimeError("dataset gone")

        enrichment.LAYER_FUNCS = [("land_cover", ok), ("vegetation", boom)]
        enrichment.CHUNK = 2
        alerts = [{"lat": 1.0 + i, "lon": 2.0} for i in range(5)]
        out, ok_layers, notes = enrichment.add_context(alerts)
        assert all(a["builtup_fraction"] == 0.1 for a in out)               # 3 chunks, all merged by index
        assert ok_layers == [l for l in enrichment.LAYERS if l not in ("vegetation",)] or "land_cover" in ok_layers
        assert "vegetation" not in ok_layers
        assert len(notes) == 1 and "vegetation" in notes[0]
    finally:
        enrichment.LAYER_FUNCS = saved
        enrichment.CHUNK = 400
        sys.modules.pop("ee", None)
