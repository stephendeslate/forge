"""Centralized logging configuration for Forge."""

from __future__ import annotations

import logging
import os


def _configure_root() -> logging.Logger:
    """Configure the forge root logger with RichHandler."""
    logger = logging.getLogger("forge")
    if logger.handlers:
        return logger

    level = os.environ.get("FORGE_LOG_LEVEL", "WARNING").upper()
    logger.setLevel(getattr(logging, level, logging.WARNING))

    try:
        from rich.console import Console as RichConsole
        from rich.logging import RichHandler

        handler = RichHandler(
            console=RichConsole(stderr=True),
            rich_tracebacks=True,
            show_path=False,
        )
    except ImportError:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))

    logger.addHandler(handler)
    return logger


_configure_root()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the forge namespace."""
    return logging.getLogger(f"forge.{name}")
