"""
TerraVision AI — persistent storage (SQLite).

Keeps every scan, every alert in it, and the analyst workflow state
(status / assignee / note / confirmed type) per alert ID. SQLite ships with
Python, so there is nothing extra to install.

Location: ./terravision.db, or set TERRAVISION_DB to a path on a persistent disk
(needed on hosts such as Streamlit Cloud whose local files are wiped on restart).
"""

from __future__ import annotations

import getpass
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

STATUSES = ["new", "under review", "verified", "false positive", "closed"]
DEFAULT_STATUS = "new"


def db_path() -> Path:
    return Path(os.environ.get("TERRAVISION_DB") or Path(__file__).with_name("terravision.db"))


@contextmanager
def _conn(path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    p = Path(path) if path else db_path()
    con = sqlite3.connect(p, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA journal_mode=WAL")
        _init(con)
        yield con
        con.commit()
    finally:
        con.close()


SCHEMA_VERSION = 3   # 1 = scans/alerts/status, 2 = + scan_runs, status_history, 3 = + scans.regional_correction


def _init(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            region        TEXT NOT NULL,
            scanned_at    TEXT NOT NULL,
            period        TEXT,
            threshold     REAL,
            min_cluster   INTEGER,
            context_ok    TEXT,
            notes         TEXT,
            resolved_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_scans_region ON scans(region, id);
        CREATE TABLE IF NOT EXISTS alerts (
            scan_id   INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
            alert_id  TEXT NOT NULL,
            category  TEXT,
            severity  TEXT,
            lifecycle TEXT,
            payload   TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_alerts_scan ON alerts(scan_id);
        CREATE TABLE IF NOT EXISTS status (
            alert_id      TEXT PRIMARY KEY,
            status        TEXT NOT NULL DEFAULT 'new',
            assignee      TEXT DEFAULT '',
            note          TEXT DEFAULT '',
            true_category TEXT DEFAULT '',
            updated_at    TEXT
        );
        """
    )
    _migrate(con)


def _migrate(con: sqlite3.Connection) -> None:
    """Bring an older database up to SCHEMA_VERSION. Each step is idempotent."""
    version = con.execute("PRAGMA user_version").fetchone()[0]
    if version < 2:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS scan_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                region      TEXT NOT NULL,
                started_at  TEXT NOT NULL,
                duration_s  REAL,
                status      TEXT NOT NULL,          -- ok | failed
                error       TEXT,
                n_alerts    INTEGER,
                period      TEXT,
                context_ok  TEXT,
                source      TEXT                    -- app | scheduled
            );
            CREATE TABLE IF NOT EXISTS status_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id      TEXT NOT NULL,
                changed_at    TEXT NOT NULL,
                actor         TEXT,
                old_status    TEXT,
                new_status    TEXT,
                assignee      TEXT,
                note          TEXT,
                true_category TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_history_alert ON status_history(alert_id, id);
            """
        )
    if version < 3:
        cols = {r[1] for r in con.execute("PRAGMA table_info(scans)")}
        if "regional_correction" not in cols:
            con.execute("ALTER TABLE scans ADD COLUMN regional_correction INTEGER")
    if version != SCHEMA_VERSION:
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


# ── scans ────────────────────────────────────────────────────────────────────

def save_scan(scan: Dict, path: Optional[Path] = None) -> int:
    """
    Persist a scan dict: {region, scanned_at, period, threshold, min_cluster,
    context_ok (list), notes (list), resolved (list), alerts (list)}.
    Returns the new scan id.
    """
    with _conn(path) as con:
        cur = con.execute(
            "INSERT INTO scans(region, scanned_at, period, threshold, min_cluster, context_ok, notes, resolved_json,"
            " regional_correction) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                scan["region"], scan["scanned_at"], scan.get("period"),
                scan.get("threshold"), scan.get("min_cluster"),
                json.dumps(scan.get("context_ok", [])), json.dumps(scan.get("notes", [])),
                json.dumps(scan.get("resolved", [])),
                None if scan.get("regional_correction") is None else int(bool(scan["regional_correction"])),
            ),
        )
        sid = cur.lastrowid
        con.executemany(
            "INSERT INTO alerts(scan_id, alert_id, category, severity, lifecycle, payload) VALUES (?,?,?,?,?,?)",
            [(sid, a["alert_id"], a.get("category"), a.get("severity"), a.get("lifecycle"),
              json.dumps(a)) for a in scan["alerts"]],
        )
        return sid


def _row_to_scan(con: sqlite3.Connection, row: sqlite3.Row, with_alerts: bool = True) -> Dict:
    scan = {
        "scan_id": row["id"], "region": row["region"], "scanned_at": row["scanned_at"],
        "period": row["period"], "threshold": row["threshold"], "min_cluster": row["min_cluster"],
        "context_ok": json.loads(row["context_ok"] or "[]"),
        "notes": json.loads(row["notes"] or "[]"),
        "resolved": json.loads(row["resolved_json"] or "[]"),
        # scans made before the correction existed are treated as "not corrected"
        "regional_correction": bool(row["regional_correction"]) if row["regional_correction"] is not None else False,
    }
    if with_alerts:
        scan["alerts"] = [
            json.loads(r["payload"])
            for r in con.execute("SELECT payload FROM alerts WHERE scan_id=? ORDER BY rowid", (row["id"],))
        ]
    return scan


def previous_scan(region: str, path: Optional[Path] = None) -> Optional[Dict]:
    """Most recent saved scan of a region (used for tracking), or None."""
    with _conn(path) as con:
        row = con.execute("SELECT * FROM scans WHERE region=? ORDER BY id DESC LIMIT 1", (region,)).fetchone()
        return _row_to_scan(con, row) if row else None


def latest_scans(path: Optional[Path] = None) -> List[Dict]:
    """The newest scan of every region, newest first, with analyst status merged into each alert."""
    with _conn(path) as con:
        rows = con.execute(
            "SELECT s.* FROM scans s JOIN (SELECT region, MAX(id) mid FROM scans GROUP BY region) m "
            "ON s.id = m.mid ORDER BY s.id DESC"
        ).fetchall()
        scans = [_row_to_scan(con, r) for r in rows]
        status = {r["alert_id"]: dict(r) for r in con.execute("SELECT * FROM status")}
    for s in scans:
        for a in s["alerts"]:
            st = status.get(a["alert_id"])
            a["status"] = st["status"] if st else DEFAULT_STATUS
            a["assignee"] = st["assignee"] if st else ""
            a["note"] = st["note"] if st else ""
            a["true_category"] = st["true_category"] if st else ""
    return scans


def alert_count(path: Optional[Path] = None) -> int:
    """Alerts in the newest scan of every region (cheap; used for the sidebar badge)."""
    with _conn(path) as con:
        row = con.execute(
            "SELECT COUNT(*) c FROM alerts WHERE scan_id IN (SELECT MAX(id) FROM scans GROUP BY region)"
        ).fetchone()
        return int(row["c"])


def scan_history(path: Optional[Path] = None) -> pd.DataFrame:
    """One row per (scan, category): how many alerts each scan found. Drives the trend chart."""
    with _conn(path) as con:
        rows = con.execute(
            "SELECT s.id scan_id, s.region, s.scanned_at, s.period, a.category, a.lifecycle, COUNT(*) n "
            "FROM scans s JOIN alerts a ON a.scan_id = s.id "
            "GROUP BY s.id, a.category, a.lifecycle ORDER BY s.id"
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows],
                        columns=["scan_id", "region", "scanned_at", "period", "category", "lifecycle", "n"])


# ── analyst workflow ─────────────────────────────────────────────────────────

def _actor() -> str:
    """Who is making a change: TERRAVISION_USER if set, else the operating-system user."""
    try:
        return os.environ.get("TERRAVISION_USER") or getpass.getuser()
    except Exception:
        return "unknown"


def set_status(alert_id: str, status: str, assignee: str = "", note: str = "",
               true_category: str = "", path: Optional[Path] = None) -> None:
    if status not in STATUSES:
        raise ValueError(f"Unknown status {status!r}")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with _conn(path) as con:
        old = con.execute("SELECT status FROM status WHERE alert_id=?", (alert_id,)).fetchone()
        con.execute(
            "INSERT INTO status(alert_id, status, assignee, note, true_category, updated_at) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(alert_id) DO UPDATE SET status=excluded.status, assignee=excluded.assignee, "
            "note=excluded.note, true_category=excluded.true_category, updated_at=excluded.updated_at",
            (alert_id, status, assignee.strip(), note.strip(), true_category, now),
        )
        # Append-only audit trail: who changed what, and when
        con.execute(
            "INSERT INTO status_history(alert_id, changed_at, actor, old_status, new_status, assignee, note, true_category)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (alert_id, now, _actor(), old["status"] if old else DEFAULT_STATUS, status,
             assignee.strip(), note.strip(), true_category),
        )


def status_history(alert_id: Optional[str] = None, limit: int = 200, path: Optional[Path] = None) -> pd.DataFrame:
    """Audit trail, newest first; for one alert or for everything."""
    q = "SELECT alert_id, changed_at, actor, old_status, new_status, assignee, note, true_category FROM status_history"
    args: tuple = ()
    if alert_id:
        q += " WHERE alert_id=?"
        args = (alert_id,)
    q += " ORDER BY id DESC LIMIT ?"
    with _conn(path) as con:
        rows = con.execute(q, args + (limit,)).fetchall()
    cols = ["alert_id", "changed_at", "actor", "old_status", "new_status", "assignee", "note", "true_category"]
    return pd.DataFrame([dict(r) for r in rows], columns=cols)


# ── scan run history (health) ────────────────────────────────────────────────

def log_run(region: str, started_at: str, duration_s: float, status: str, error: str = "",
            n_alerts: int = 0, period: str = "", context_ok: Optional[List[str]] = None,
            source: str = "app", path: Optional[Path] = None) -> None:
    """Record one scan attempt, successful or not, so failures are visible later."""
    with _conn(path) as con:
        con.execute(
            "INSERT INTO scan_runs(region, started_at, duration_s, status, error, n_alerts, period, context_ok, source)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (region, started_at, round(duration_s, 1), status, error[:500], n_alerts, period,
             json.dumps(context_ok or []), source),
        )


def recent_runs(limit: int = 15, path: Optional[Path] = None) -> pd.DataFrame:
    cols = ["started_at", "region", "status", "duration_s", "n_alerts", "period", "context_ok", "source", "error"]
    with _conn(path) as con:
        rows = con.execute(
            "SELECT started_at, region, status, duration_s, n_alerts, period, context_ok, source, error "
            "FROM scan_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    df = pd.DataFrame([dict(r) for r in rows], columns=cols)
    if not df.empty:
        df["context_ok"] = df["context_ok"].map(lambda j: ", ".join(json.loads(j or "[]")) or "—")
    return df


def set_status_many(changes: List[Dict], path: Optional[Path] = None) -> int:
    for c in changes:
        set_status(c["alert_id"], c["status"], c.get("assignee", ""), c.get("note", ""),
                   c.get("true_category", ""), path)
    return len(changes)


# ── one-time import of the earlier JSON-file store ──────────────────────────

def import_legacy_json(scan_dir: Optional[Path] = None, path: Optional[Path] = None) -> int:
    """Import ./scans/*.json (written by the previous version) for regions not yet in the database."""
    scan_dir = Path(scan_dir) if scan_dir else Path(__file__).with_name("scans")
    if not scan_dir.exists():
        return 0
    imported = 0
    for p in scan_dir.glob("*.json"):
        try:
            d = json.loads(p.read_text())
            if not (d.get("region") and isinstance(d.get("alerts"), list)):
                continue
            if previous_scan(d["region"], path):
                continue
            for a in d["alerts"]:
                a.setdefault("lifecycle", "new")
                a.setdefault("times_seen", 1)
                a.setdefault("first_seen", d.get("scanned_at"))
            save_scan({"region": d["region"], "scanned_at": d.get("scanned_at", ""),
                       "period": None, "alerts": d["alerts"]}, path)
            imported += 1
        except (OSError, ValueError, KeyError):
            continue
    return imported
