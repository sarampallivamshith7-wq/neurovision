"""Logging utilities for NeuroVision."""

import logging
import sys


def get_logger(name: str = "neurovision", level: int = logging.INFO) -> logging.Logger:
    """Return a formatted logger writing to sys.stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger
