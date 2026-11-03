"""
TerraVision AI — reliability helpers.

* call_with_retry: re-run a flaky call (Earth Engine quota / network hiccups) with
  exponential backoff, but fail immediately on permanent errors such as a missing
  asset or a permission problem, so real bugs are never hidden behind retries.
* logger: one rotating log file (logs/terravision.log) plus the console.
"""

from __future__ import annotations

import logging
import re
import logging.handlers
import time
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")

TRANSIENT_MARKERS = (
    "quota", "too many requests", "too many concurrent", "capacity exceeded", "rate limit", "timed out", "timeout", "temporarily",
    "internal error", "unavailable", "connection reset", "connection aborted",
    "connection refused", "remote end closed", "broken pipe", "max retries exceeded", "sslerror",
    "user memory limit",
)
# HTTP status codes only count when they look like one ("HTTP 503", "status code: 429"),
# so a number such as "scale 500" in an unrelated message is not mistaken for a server error.
TRANSIENT_STATUS = re.compile(r"(http|status|error|code)\W{0,4}(429|500|502|503|504)\b")
PERMANENT_MARKERS = (
    "not found", "does not exist", "permission denied", "not authorized", "invalid argument",
    "http 401", "http 403", "http 404", "authenticat", "credentials",
)


def is_transient(error: BaseException) -> bool:
    """True when retrying has a realistic chance of helping."""
    text = f"{type(error).__name__} {error}".lower()
    if any(m in text for m in PERMANENT_MARKERS):
        return False
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    return any(m in text for m in TRANSIENT_MARKERS) or bool(TRANSIENT_STATUS.search(text))


def call_with_retry(
    fn: Callable[..., T],
    *args,
    attempts: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs,
) -> T:
    """Call fn(*args, **kwargs); on a transient error wait 2s, 4s, 8s… and try again."""
    log = get_logger()
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == attempts or not is_transient(e):
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            log.warning("Transient error (%s: %s); retry %d/%d in %.0fs",
                        type(e).__name__, str(e)[:120], attempt, attempts - 1, delay)
            sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


_LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    """Shared logger; safe to call repeatedly (handlers are added once)."""
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    log = logging.getLogger("terravision")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        log.addHandler(console)
        try:
            log_dir = Path(__file__).with_name("logs")
            log_dir.mkdir(exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                log_dir / "terravision.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
            fh.setFormatter(fmt)
            log.addHandler(fh)
        except OSError:
            pass   # read-only disk: console logging is enough
    log.propagate = False
    _LOGGER = log
    return log
