"""Logging utilities with rich formatting."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Custom theme for fingerprint-squared
FP2_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "metric": "magenta",
    "model": "blue",
    "bias": "red",
    "fairness": "green",
})

console = Console(theme=FP2_THEME)

_loggers: dict[str, logging.Logger] = {}


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    rich_tracebacks: bool = True,
) -> None:
    """
    Set up logging with rich formatting.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        rich_tracebacks: Whether to use rich tracebacks
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    handlers = [
        RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            rich_tracebacks=rich_tracebacks,
            tracebacks_show_locals=True,
        )
    ]

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the given name."""
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]


class ProgressLogger:
    """Context manager for logging progress of evaluation tasks."""

    def __init__(
        self,
        description: str,
        total: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.description = description
        self.total = total
        self.logger = logger or get_logger("fingerprint_squared")
        self.current = 0

    def __enter__(self):
        self.logger.info(f"Starting: {self.description}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.info(f"Completed: {self.description}")
        else:
            self.logger.error(f"Failed: {self.description} - {exc_val}")
        return False

    def update(self, n: int = 1, message: Optional[str] = None):
        """Update progress."""
        self.current += n
        if message and self.total:
            self.logger.debug(f"[{self.current}/{self.total}] {message}")


def log_metric(name: str, value: float, context: Optional[str] = None):
    """Log a metric value with special formatting."""
    logger = get_logger("fingerprint_squared.metrics")
    ctx = f" ({context})" if context else ""
    logger.info(f"[metric]{name}{ctx}: {value:.4f}[/metric]")


def log_bias_detection(
    bias_type: str,
    severity: str,
    details: str,
):
    """Log a detected bias with severity."""
    logger = get_logger("fingerprint_squared.bias")
    severity_colors = {
        "low": "yellow",
        "medium": "orange",
        "high": "red",
        "critical": "bold red",
    }
    color = severity_colors.get(severity.lower(), "white")
    logger.warning(f"[{color}]BIAS DETECTED[/{color}] [{bias_type}] {severity.upper()}: {details}")
