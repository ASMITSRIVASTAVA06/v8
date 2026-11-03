import sqlite3
import tempfile
from pathlib import Path

import store


def test_old_database_is_upgraded_in_place():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "old.db"
        con = sqlite3.connect(db)          # a v1 database: has the old tables, user_version 0
        con.executescript("CREATE TABLE scans(id INTEGER PRIMARY KEY AUTOINCREMENT, region TEXT NOT NULL, "
                          "scanned_at TEXT NOT NULL, period TEXT, threshold REAL, min_cluster INTEGER, "
                          "context_ok TEXT, notes TEXT, resolved_json TEXT);")
        con.execute("INSERT INTO scans(region, scanned_at) VALUES ('R', '2026-01-01 00:00')")
        con.commit(); con.close()
        assert store.previous_scan("R", db)["region"] == "R"          # old data still readable
        store.log_run("R", "2026-02-01 10:00", 12.3, "ok", path=db)   # new tables exist now
        con = sqlite3.connect(db)
        assert con.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION
        con.close()


def test_run_history_records_failures():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        store.log_run("A", "2026-02-01 10:00", 5.0, "ok", n_alerts=7, period="2026-01", context_ok=["land_cover"], path=db)
        store.log_run("B", "2026-02-02 10:00", 1.0, "failed", error="boom", path=db)
        runs = store.recent_runs(path=db)
        assert list(runs["status"]) == ["failed", "ok"]                # newest first
        assert runs.iloc[1]["context_ok"] == "land_cover" and runs.iloc[0]["error"] == "boom"


def test_review_changes_leave_an_audit_trail():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.db"
        store.set_status("TV-1", "under review", assignee="Ramesh", path=db)
        store.set_status("TV-1", "verified", note="3 huts", path=db)
        store.set_status("TV-2", "false positive", path=db)
        h = store.status_history("TV-1", path=db)
        assert list(h["new_status"]) == ["verified", "under review"]   # newest first
        assert list(h["old_status"]) == ["under review", "new"]
        assert h["actor"].iloc[0]
        assert len(store.status_history(path=db)) == 3
