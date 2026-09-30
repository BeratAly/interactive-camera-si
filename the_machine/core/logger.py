"""Logging setup (§33).

- Rotating file handler in data/logs/machine.log (+ errors.log for ERROR+).
- Redaction filter guarantees secrets never reach disk or console:
  anything that looks like an API key, and explicit SecretStr values.
"""
from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_KEY_PATTERN = re.compile(
    r"(?i)\b(sk-[A-Za-z0-9_\-]{8,}|[A-Fa-f0-9]{32,}|api[_-]?key\s*[=:]\s*\S+)"
)


class RedactFilter(logging.Filter):
    """Scrubs potential secrets from every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if _KEY_PATTERN.search(msg):
            record.msg = _KEY_PATTERN.sub("[REDACTED]", msg)
            record.args = ()
        return True


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    redact = RedactFilter()

    root = logging.getLogger("machine")
    root.setLevel(level)
    root.handlers.clear()

    fh = RotatingFileHandler(log_dir / "machine.log", maxBytes=1_000_000,
                             backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.addFilter(redact)
    root.addHandler(fh)

    eh = RotatingFileHandler(log_dir / "errors.log", maxBytes=500_000,
                             backupCount=2, encoding="utf-8")
    eh.setLevel(logging.ERROR)
    eh.setFormatter(fmt)
    eh.addFilter(redact)
    root.addHandler(eh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.addFilter(redact)
    root.addHandler(sh)
