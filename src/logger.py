"""Structured logging infrastructure for Kizuna Privacy Engine.

Provides structured logging with JSON output, correlation IDs, log rotation,
and PII redaction capabilities.
"""

import logging
import os
import re
import sys
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.typing import EventDict, WrappedLogger

# PII patterns to redact from logs
PII_PATTERNS = [
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP_REDACTED]"),  # IP addresses
    (re.compile(r"[a-zA-Z]:[/\\]Users[/\\][^/\\]+"), "[USER_PATH]"),  # User paths (Windows/Unix)
    (re.compile(r"/home/[^/]+"), "/home/[USER]"),  # Linux user paths
    (re.compile(r"/Users/[^/]+"), "/Users/[USER]"),  # macOS user paths
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "[EMAIL_REDACTED]",
    ),  # Emails
]


def redact_pii(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Redact PII from log events.

    Args:
        logger: Wrapped logger instance
        method_name: Name of the method being logged
        event_dict: Event dictionary to process

    Returns:
        Event dictionary with PII redacted
    """
    for key, value in event_dict.items():
        if isinstance(value, str):
            for pattern, replacement in PII_PATTERNS:
                value = pattern.sub(replacement, value)
            event_dict[key] = value
    return event_dict


def add_correlation_id(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add correlation ID to log events.

    Args:
        logger: Wrapped logger instance
        method_name: Name of the method being logged
        event_dict: Event dictionary to process

    Returns:
        Event dictionary with correlation_id added
    """
    # Get correlation ID from thread-local storage or generate new one
    if "correlation_id" not in event_dict:
        event_dict["correlation_id"] = str(uuid.uuid4())
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    component: str = "kizuna",
    enable_correlation_id: bool = True,
    enable_pii_redaction: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> structlog.BoundLogger:
    """Set up structured logging with rotation and PII redaction.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        component: Component name for the logger
        enable_correlation_id: Whether to add correlation IDs
        enable_pii_redaction: Whether to redact PII from logs
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured structlog logger
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Configure Python standard logging
    handlers = []

    # Console handler (human-readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    handlers.append(console_handler)

    # File handler with rotation (JSON for machine parsing)
    log_file = log_path / f"{component}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
    )

    # Build structlog processor chain
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Add correlation ID processor
    if enable_correlation_id:
        processors.append(add_correlation_id)

    # Add PII redaction processor
    if enable_pii_redaction:
        processors.append(redact_pii)

    # Add final renderer
    # Console: colorful key-value output
    # File: JSON for machine parsing
    processors.append(structlog.stdlib.ProcessorFormatter.wrap_for_formatter)

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure formatters for handlers
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True),
    )
    console_handler.setFormatter(console_formatter)

    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
    )
    file_handler.setFormatter(file_formatter)

    # Return configured logger for the component
    return structlog.get_logger(component)


def get_logger(
    component: str,
    log_level: Optional[str] = None,
    log_dir: Optional[str] = None,
    **kwargs: Any,
) -> structlog.BoundLogger:
    """Get a logger for a specific component.

    Args:
        component: Component name (e.g., 'ingestion', 'engine', 'privacy')
        log_level: Override log level (uses environment or INFO if not set)
        log_dir: Override log directory (uses environment or 'logs' if not set)
        **kwargs: Additional kwargs passed to setup_logging

    Returns:
        Configured structlog logger for the component
    """
    log_level = log_level or os.getenv("KIZUNA_LOG_LEVEL", "INFO")
    log_dir = log_dir or os.getenv("KIZUNA_LOG_DIR", "logs")

    return setup_logging(
        log_level=log_level,
        log_dir=log_dir,
        component=component,
        **kwargs,
    )


# Pre-configured loggers for common components
def get_ingestion_logger() -> structlog.BoundLogger:
    """Get logger for ingestion components."""
    return get_logger("ingestion")


def get_engine_logger() -> structlog.BoundLogger:
    """Get logger for embedding engine."""
    return get_logger("engine")


def get_privacy_logger() -> structlog.BoundLogger:
    """Get logger for privacy layer."""
    return get_logger("privacy")


def get_database_logger() -> structlog.BoundLogger:
    """Get logger for database operations."""
    return get_logger("database")


def get_anomaly_logger() -> structlog.BoundLogger:
    """Get logger for anomaly detection."""
    return get_logger("anomaly")


def get_dashboard_logger() -> structlog.BoundLogger:
    """Get logger for dashboard."""
    return get_logger("dashboard")


# Context manager for correlation ID
class CorrelationContext:
    """Context manager for setting correlation ID across log messages."""

    def __init__(self, correlation_id: Optional[str] = None):
        """Initialize correlation context.

        Args:
            correlation_id: Correlation ID to use (generates new if None)
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self._token: Optional[object] = None

    def __enter__(self) -> str:
        """Enter correlation context."""
        self._token = structlog.contextvars.bind_contextvars(correlation_id=self.correlation_id)
        return self.correlation_id

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit correlation context."""
        if self._token is not None:
            structlog.contextvars.unbind_contextvars("correlation_id")
