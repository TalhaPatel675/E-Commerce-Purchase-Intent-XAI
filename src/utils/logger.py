"""Centralised logging (PRD §5)."""

from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    level_name = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    level_value = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level_value)

    formatter = logging.Formatter(_DEFAULT_FORMAT)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
