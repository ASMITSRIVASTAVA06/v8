"""
TerraVision AI — scheduled weather and cyclone watch.

Checks the forecast and active cyclones for the forest regions and, only when something at or above
the chosen level appears (and has not already been announced), sends an alert by e-mail and/or Telegram.

  python scheduled_weather.py --regions all --min-level 2 --dry-run
  python scheduled_weather.py --regions "Western Ghats, India;Sundarbans, Bangladesh" --telegram

Run it every few hours (cron: 0 */6 * * *  or the GitHub Actions workflow scheduled_weather.yml).
Settings come from .env / environment: SMTP_*, DIGEST_RECIPIENTS, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import cyclones
import riskzones
import weather_alerts
from alert_tools import parse_recipients
from regions import REGION_BOUNDS
from scheduled_scan import parse_regions


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Weather and cyclone watch for forest regions.")
    p.add_argument("--regions", default="all", help="'all' or region names separated by ';'")
    p.add_argument("--hours", type=int, default=72, help="forecast horizon in hours (24-168)")
    p.add_argument("--min-level", type=int, default=2, choices=[1, 2, 3], help="1 Watch, 2 Warning, 3 Severe")
    p.add_argument("--to", default=os.environ.get("DIGEST_RECIPIENTS", ""), help="e-mail recipients, comma-separated")
    p.add_argument("--telegram", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the alert instead of sending it")
    p.add_argument("--resend", action="store_true", help="ignore the record of what was already sent")
    args = p.parse_args(argv)

    regions = parse_regions(args.regions)
    recipients, invalid = parse_recipients(args.to)
    if invalid:
        print(f"Ignoring invalid recipients: {invalid}")

    try:
        storms, notes = cyclones.list_active_cyclones()
        print(f"{len(storms)} active cyclone(s) worldwide")
    except cyclones.CycloneError as e:          # the forecast check is still worth running
        storms, notes = [], [f"Cyclone feed unavailable: {e}"]
    for n in notes:
        print("  note:", n)

    rows = riskzones.region_overview(storms, hours=max(24, min(168, args.hours)), regions=regions)
    for r in rows:
        lvl = "n/a" if r["level"] is None else weather_alerts.weather.LEVELS[r["level"]]
        print(f"  {lvl:<8} {r['region']}: {r['headline']}")
    if all(r["level"] is None for r in rows):
        print("No forecast could be fetched for any region.")
        return 1

    near = [s for s in storms if any(cyclones.worst_cyclone_threat(*_centre(r["region"]), [s])["level"] > 0 for r in rows)]
    message = weather_alerts.build_message(rows, near, args.min_level)
    if message is None:
        print(f"\nNothing at or above level {args.min_level}: no alert sent.")
        return 0
    state = weather_alerts.load_state()
    if not args.resend and not args.dry_run and not weather_alerts.should_send(message, state):
        print("\nSame situation as the last alert: not repeating it.")
        return 0
    for line in weather_alerts.deliver(message, recipients, args.telegram, args.dry_run):
        print(" ", line)
    if not args.dry_run and (recipients or args.telegram):
        weather_alerts.save_state(message)
    elif not (recipients or args.telegram) and not args.dry_run:
        print("  No recipients configured (use --to / DIGEST_RECIPIENTS and/or --telegram).")
    return 0


def _centre(region: str):
    min_lon, min_lat, max_lon, max_lat = REGION_BOUNDS[region]
    return (min_lat + max_lat) / 2, (min_lon + max_lon) / 2


if __name__ == "__main__":
    sys.exit(main())
