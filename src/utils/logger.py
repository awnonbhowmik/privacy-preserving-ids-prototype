"""
logger.py — Centralised logging configuration.

Every module in this project calls get_logger(__name__) to get a consistent
logger. This means all log output shares the same format and can be directed
to a file or stdout from one place.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Processing started")
"""

import logging
import sys
from pathlib import Path


# ── Log format ─────────────────────────────────────────────────────────────────
# Timestamp | level (padded) | module name | message
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    If the logger already has handlers attached (i.e., it was already
    configured in a previous call), this function returns it unchanged.
    This prevents duplicate log lines when modules are re-imported.

    Args:
        name:     Typically pass __name__ from the calling module so log
                  lines identify their source (e.g. src.data.preprocessor).
        level:    Logging level. Defaults to INFO. Use logging.DEBUG during
                  development for verbose chunk-level output.
        log_file: Optional path to write logs to a file in addition to stdout.
                  Useful for capturing long preprocessing runs.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Guard: avoid adding duplicate handlers if called more than once
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    # Always write to stdout so Streamlit and terminal both show logs
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Optionally mirror to a log file (useful for long preprocessing runs)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent log records from propagating to the root logger, which would
    # cause duplicate output if the root logger is also configured.
    logger.propagate = False

    return logger
