"""Logging configuration for particlefilterbox."""

import logging

_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the particlefilterbox namespace."""
    logger = logging.getLogger(f"particlefilterbox.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
    return logger
