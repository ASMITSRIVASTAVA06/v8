import tempfile
from pathlib import Path

import store


def scan(region="Aravalli Hills, India", ids=("TV-1", "TV-2"), when="2026-02-01 10:00"):
    return {"region": region, "scanned_at": when, "period": "2026-01", "threshold": 5.0, "min_cluster": 3,
            "context_ok": ["land_cover"], "notes": ["n"], "resolved": [{"alert_id": "TV-0"}],
            "alerts": [{"alert_id": i, "category": "illegal_encroachment", "severity": "high",
                        "lifecycle": "new", "lat": 26.0, "lon": 75.0, "confidence": 70} for i in ids]}


def test_save_and_reload_latest_per_region():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        store.save_scan(scan(ids=("TV-1",)), db)
        store.save_scan(scan(ids=("TV-1", "TV-2"), when="2026-03-01 10:00"), db)
        store.save_scan(scan(region="Borneo Rainforest", ids=("TV-9",)), db)
        latest = {s["region"]: s for s in store.latest_scans(db)}
        assert len(latest) == 2
        assert len(latest["Aravalli Hills, India"]["alerts"]) == 2          # newest scan only
        assert latest["Aravalli Hills, India"]["resolved"] == [{"alert_id": "TV-0"}]
        assert store.previous_scan("Borneo Rainforest", db)["alerts"][0]["alert_id"] == "TV-9"
        assert store.previous_scan("Nowhere", db) is None


def test_status_workflow_is_merged_and_survives_new_scans():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        store.save_scan(scan(), db)
        store.set_status("TV-1", "verified", "Ravi", "checked on site", "illegal_encroachment", db)
        store.save_scan(scan(when="2026-03-01 10:00"), db)                   # a later scan, same IDs
        alerts = {a["alert_id"]: a for a in store.latest_scans(db)[0]["alerts"]}
        assert alerts["TV-1"]["status"] == "verified" and alerts["TV-1"]["assignee"] == "Ravi"
        assert alerts["TV-2"]["status"] == "new"


def test_invalid_status_rejected():
    with tempfile.TemporaryDirectory() as d:
        try:
            store.set_status("TV-1", "bogus", path=Path(d) / "t.db")
        except ValueError:
            return
        raise AssertionError("expected ValueError")


def test_history_counts():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        store.save_scan(scan(), db)
        h = store.scan_history(db)
        assert int(h["n"].sum()) == 2 and set(h["category"]) == {"illegal_encroachment"}


def test_legacy_json_import():
    import json
    with tempfile.TemporaryDirectory() as d:
        sd = Path(d) / "scans"; sd.mkdir()
        (sd / "x.json").write_text(json.dumps({"region": "Borneo Rainforest", "scanned_at": "2026-01-01 00:00",
            "alerts": [{"alert_id": "TV-7", "category": "anomaly", "severity": "medium", "lat": 1, "lon": 2}]}))
        db = Path(d) / "t.db"
        assert store.import_legacy_json(sd, db) == 1
        assert store.import_legacy_json(sd, db) == 0                          # idempotent
        assert store.latest_scans(db)[0]["alerts"][0]["lifecycle"] == "new"
