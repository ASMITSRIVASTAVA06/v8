"""
TerraVision AI — weather and cyclone alerts.

Builds a plain-language message from the flagged zones and storms, sends it by e-mail and/or
Telegram using the same settings as the forest alerts, and remembers what it already told people so
the same warning is not repeated every hour.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import weather
from alert_tools import (
    get_smtp_config,
    get_telegram_config,
    send_alert_email,
    send_telegram,
)

def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


STATE_PATH = Path(os.environ.get("TERRAVISION_WEATHER_STATE", Path(__file__).parent / "data_cache" / "weather_state.json"))
DISCLAIMER = ("This is an automated indication from model forecasts and public cyclone advisories, not an "
              "official warning. Follow your national weather service (IMD, NHC, JTWC) for decisions.")


def build_message(rows: List[Dict], storms: List[Dict], min_level: int = 2) -> Optional[Dict]:
    """
    {subject, body, signature, level} for regions at/above `min_level` and any storm near them,
    or None when nothing needs saying. `rows` are region_overview rows.
    """
    hot = [r for r in rows if (r.get("level") or 0) >= min_level]
    if not hot:
        return None
    top = max(r["level"] for r in hot)
    lines = []
    for r in hot:
        when = f" · around {r['peak_at'].replace('T', ' ')} UTC" if r.get("peak_at") else ""
        lines.append(f"{weather.LEVELS[r['level']].upper()} · {r['region']}: {r['headline']} "
                     f"({r['flagged']} of {r['cells']} zones at Warning or above){when}")
    storm_lines = [f"{s['name']} ({s['alert']} alert, {s['severity_text'] or 'tropical cyclone'})" for s in storms]
    body = ("TerraVision weather watch\n\n" + "\n".join(lines)
            + (("\n\nActive cyclones: " + "; ".join(storm_lines)) if storm_lines else "")
            + f"\n\nIssued {_utcnow():%Y-%m-%d %H:%M} UTC.\n{DISCLAIMER}")
    # the signature ignores wording and times so only a genuine change in who is at risk re-sends
    sig_src = json.dumps(sorted((r["region"], r["level"]) for r in hot) + sorted(s["name"] for s in storms))
    return {"subject": f"TerraVision: {weather.LEVELS[top]}-level weather risk in {len(hot)} forest region(s)",
            "body": body, "signature": hashlib.sha1(sig_src.encode()).hexdigest()[:16], "level": top}


def load_state(path: Path = STATE_PATH) -> Dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def should_send(message: Dict, state: Dict, now: Optional[datetime] = None, repeat_hours: int = 24) -> bool:
    """True if this signature has not been sent, or was sent more than `repeat_hours` ago and is Severe."""
    last = state.get("signature"), state.get("sent_at")
    if last[0] != message["signature"]:
        return True
    try:
        age = ((now or _utcnow()) - datetime.strptime(last[1], "%Y-%m-%d %H:%M")).total_seconds() / 3600
    except (TypeError, ValueError):
        return True
    return message["level"] >= 3 and age >= repeat_hours


def save_state(message: Dict, path: Path = STATE_PATH, now: Optional[datetime] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"signature": message["signature"],
                                      "sent_at": (now or _utcnow()).strftime("%Y-%m-%d %H:%M")}))


def deliver(message: Dict, email_to: List[str], use_telegram: bool, dry_run: bool = False) -> List[str]:
    log: List[str] = []
    if dry_run:
        return ["[dry run] nothing sent", message["subject"], "", message["body"]]
    if email_to:
        smtp = get_smtp_config()
        if smtp is None:
            log.append("E-mail skipped: SMTP_HOST / SMTP_USER / SMTP_PASSWORD are not set.")
        else:
            try:
                send_alert_email(smtp, email_to, message["subject"], message["body"], {})
                log.append(f"E-mailed weather alert to {', '.join(email_to)}")
            except Exception as e:
                log.append(f"E-mail FAILED: {e}")
    if use_telegram:
        tg = get_telegram_config()
        if tg is None:
            log.append("Telegram skipped: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set.")
        else:
            try:
                send_telegram(tg, message["subject"] + "\n\n" + message["body"])
                log.append("Sent weather alert to Telegram")
            except Exception as e:
                log.append(f"Telegram FAILED: {e}")
    return log
