"""
TerraVision AI — scheduled scan with alert digest.

Runs the same pipeline as the web app, headless, for one or more regions, saves
the results to the database, and — only when something NEW and credible turned
up — sends a digest by e-mail and/or Telegram.

Examples
--------
  python scheduled_scan.py --regions "Aravalli Hills, India"
  python scheduled_scan.py --regions all --min-confidence 65
  python scheduled_scan.py --regions "Borneo Rainforest" --dry-run     # scan, print digest, send nothing

Schedule it (run weekly or monthly; VIIRS publishes monthly composites):
  cron:            0 6 * * MON  cd /path/to/tvai_without && python scheduled_scan.py --regions all
  GitHub Actions:  see .github/workflows/scheduled_scan.yml

Settings come from .env / environment: GEE_PROJECT_ID (+ GEE_SERVICE_ACCOUNT_JSON when headless),
SMTP_*, DIGEST_RECIPIENTS (comma-separated), TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TERRAVISION_DB.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

from alert_tools import (
    build_html_report,
    build_summary_text,
    get_smtp_config,
    get_telegram_config,
    parse_recipients,
    send_alert_email,
    send_telegram,
    to_csv_bytes,
    to_dataframe,
)
from evidence import zone_of
from regions import REGION_BOUNDS


# ── choosing what is worth a notification ───────────────────────────────────

def select_digest_alerts(
    alerts: List[Dict],
    min_confidence: int = 60,
    only_new: bool = True,
    categories: Tuple[str, ...] = ("illegal_encroachment",),
) -> List[Dict]:
    """
    Alerts worth pushing to people: new (or all, if only_new is False), of the wanted
    types, at or above the confidence bar, and not judged a likely false positive.
    """
    out = []
    for a in alerts:
        if categories and a.get("category") not in categories:
            continue
        if only_new and a.get("lifecycle") != "new":
            continue
        if a.get("confidence", 0) < min_confidence:
            continue
        if a.get("verdict") == "Likely false positive" or a.get("urban_light"):
            continue
        if (a.get("zone") or zone_of(a)) == "outside":
            continue
        if a.get("status") in ("verified", "false positive", "closed"):
            continue
        out.append(a)
    return out


def build_digest(scans: List[Dict], **select_kwargs) -> Optional[Dict]:
    """
    Combine the selected alerts from several region scans into one digest, or return
    None when there is nothing to report. Result: {subject, body, df, count, regions}.
    """
    picked: List[Dict] = []
    for s in scans:
        picked += select_digest_alerts(s["alerts"], **select_kwargs)
    if not picked:
        return None
    df = to_dataframe(picked)
    regions = sorted(df["region"].unique())
    picked_ids = {id(a) for a in picked}
    scan_lines = []
    for s in scans:
        n = sum(1 for a in s["alerts"] if id(a) in picked_ids)
        scan_lines.append(f"{s['region']}: {n} to review (period {s.get('period') or 'unknown'}, "
                          f"{len(s.get('resolved', []))} resolved)")
    body = (
        "TerraVision found new alerts that pass the evidence checks.\n\n"
        + "\n".join(scan_lines) + "\n\n" + build_summary_text(df, title="New alerts to verify", top_n=12)
    )
    subject = f"TerraVision: {len(picked)} new alert(s) to verify — {', '.join(regions)}"
    return {"subject": subject, "body": body, "df": df, "count": len(picked), "regions": regions}


# ── delivery ─────────────────────────────────────────────────────────────────

def deliver(digest: Dict, email_to: List[str], use_telegram: bool, dry_run: bool = False) -> List[str]:
    """Send the digest. Returns human-readable lines describing what happened."""
    log: List[str] = []
    df = digest["df"]

    if dry_run:
        return ["[dry run] nothing sent", digest["subject"], "", digest["body"]]

    if email_to:
        smtp = get_smtp_config()
        if smtp is None:
            log.append("E-mail skipped: SMTP_HOST / SMTP_USER / SMTP_PASSWORD are not set.")
        else:
            try:
                send_alert_email(
                    smtp, email_to, digest["subject"], digest["body"],
                    {"terravision_new_alerts.csv": (to_csv_bytes(df), "text/csv"),
                     "terravision_report.html": (build_html_report(df), "text/html")},
                )
                log.append(f"E-mailed digest to {', '.join(email_to)}")
            except Exception as e:
                log.append(f"E-mail FAILED: {e}")

    if use_telegram:
        tg = get_telegram_config()
        if tg is None:
            log.append("Telegram skipped: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set.")
        else:
            try:
                send_telegram(tg, digest["subject"] + "\n\n" + build_summary_text(df, top_n=8))
                log.append("Sent digest to Telegram")
            except Exception as e:
                log.append(f"Telegram FAILED: {e}")
    return log


# ── main ─────────────────────────────────────────────────────────────────────

def parse_regions(text: str) -> List[str]:
    if text.strip().lower() == "all":
        return list(REGION_BOUNDS)
    names = [t.strip() for t in text.split(";") if t.strip()] if ";" in text else [text.strip()]
    unknown = [n for n in names if n not in REGION_BOUNDS]
    if unknown:
        raise SystemExit(f"Unknown region(s): {unknown}. Choose from: {list(REGION_BOUNDS)} (or 'all').\n"
                         "Separate several regions with ';' because names contain commas.")
    return names


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Scan regions headless and send a digest of new alerts.")
    p.add_argument("--regions", required=True, help="'all' or region names separated by ';'")
    p.add_argument("--threshold", type=float, default=5.0, help="alert threshold (nW/cm²/sr)")
    p.add_argument("--min-cluster", type=int, default=3)
    p.add_argument("--no-context", action="store_true", help="skip cross-checks with other satellites")
    p.add_argument("--no-regional-correction", action="store_true",
                   help="do not remove the district-wide brightening trend before detecting clusters")
    p.add_argument("--refresh", action="store_true", help="re-download satellite rasters")
    p.add_argument("--min-confidence", type=int, default=60, help="only notify at/above this confidence")
    p.add_argument("--include-recurring", action="store_true", help="also notify about recurring alerts")
    p.add_argument("--to", default=os.environ.get("DIGEST_RECIPIENTS", ""), help="e-mail recipients, comma-separated")
    p.add_argument("--telegram", action="store_true", help="also post the digest to Telegram")
    p.add_argument("--dry-run", action="store_true", help="print the digest instead of sending it")
    args = p.parse_args(argv)

    from pipeline import run_scan   # imported here so --help works without Earth Engine installed

    regions = parse_regions(args.regions)
    recipients, invalid = parse_recipients(args.to)
    if invalid:
        print(f"Ignoring invalid recipients: {invalid}")

    scans, failures = [], 0
    for region in regions:
        print(f"\n=== {region} ===")
        try:
            scan = run_scan(region, threshold=args.threshold, min_cluster=args.min_cluster,
                            enrich=not args.no_context, refresh=args.refresh, source="scheduled",
                            regional_correction=not args.no_regional_correction,
                            progress=lambda m: print("  ", m))
            print(f"  {len(scan['alerts'])} alerts "
                  f"({sum(a.get('lifecycle') == 'new' for a in scan['alerts'])} new, "
                  f"{len(scan.get('resolved', []))} resolved)")
            for n in scan.get("notes", []):
                print("  note:", n)
            scans.append(scan)
        except Exception as e:      # one failing region must not stop the others
            failures += 1
            print(f"  FAILED: {e}")

    digest = build_digest(scans, min_confidence=args.min_confidence, only_new=not args.include_recurring)
    if digest is None:
        print("\nNothing new above the confidence bar — no digest sent.")
    else:
        print(f"\nDigest: {digest['count']} alert(s)")
        for line in deliver(digest, recipients, args.telegram, args.dry_run):
            print(" ", line)
        if not (recipients or args.telegram) and not args.dry_run:
            print("  No recipients configured (use --to / DIGEST_RECIPIENTS and/or --telegram).")

    return 1 if regions and failures == len(regions) else 0


if __name__ == "__main__":
    sys.exit(main())
