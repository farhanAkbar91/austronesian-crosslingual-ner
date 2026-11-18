"""Logging helpers: one shared configuration, loggers per module."""

from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str = "axner") -> logging.Logger:
    """Return a logger; configure the root 'axner' logger once on first use."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
