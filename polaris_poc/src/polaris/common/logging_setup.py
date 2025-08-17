import logging
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

from enum import Enum

from polaris.common.config import load_config, get_config

load_config(search_paths=[Path("/home/divyansh/serc/POLARIS/polaris_poc/src/config/polaris_config.yaml")],
            required_keys=["LOGGER_NAME", "LOGGER_LEVEL", "LOGGER_FORMAT"])

class LogFormat(Enum):
    PRETTY = "pretty"
    JSON = "json"


# ANSI color codes
RESET = "\033[0m"
COLORS = {
    "DEBUG": "\033[37m",   # White
    "INFO": "\033[36m",    # Cyan
    "WARNING": "\033[33m", # Yellow
    "ERROR": "\033[31m",   # Red
    "CRITICAL": "\033[41m\033[97m",  # White on Red background
    "TIME": "\033[90m",    # Gray for timestamps
    "MODULE": "\033[35m",  # Magenta
    "MESSAGE": "\033[0m",  # Default
}


# ----------------------------- Utilities & Logging -----------------------------

class PrettyColoredFormatter(logging.Formatter):
    """
    Pretty, human-readable, colored log formatter.
    Format:
    2025-08-13 14:35:12.345 UTC | INFO     | monitor_adapter:123 | publisher_stopped
    """

    def format(self, record: logging.LogRecord) -> str:
        # Timestamp in UTC
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        timestamp_colored = f"{COLORS['TIME']}{timestamp} UTC{RESET}"

        # Level name with color
        level_color = COLORS.get(record.levelname, "")
        level_name_colored = f"{level_color}{record.levelname:<8}{RESET}"

        # Module + line number
        location_colored = f"{COLORS['MODULE']}{record.module}:{record.lineno}{RESET}"

        # Main message
        message_colored = f"{COLORS['MESSAGE']}{record.getMessage()}{RESET}"

        # Append exception info if present
        if record.exc_info:
            message_colored += "\n" + self.formatException(record.exc_info)

        return f"{timestamp_colored} | {level_name_colored} | {location_colored} | {message_colored}"

class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for structured logs."""
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge extra dict-like attributes, if present
        for k, v in record.__dict__.items():
            if k in ("args", "msg", "levelname", "levelno", "pathname", "filename",
                     "module", "exc_info", "exc_text", "stack_info", "lineno",
                     "funcName", "created", "msecs", "relativeCreated", "thread",
                     "threadName", "processName", "process"):  # skip std keys
                continue
            # only include JSON-serializable-ish values
            try:
                json.dumps(v)
                base[k] = v
            except Exception:
                base[k] = str(v)
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, separators=(",", ":"))


def setup_logging() -> logging.Logger:
    logger = logging.getLogger(get_config("LOGGER_NAME", "polaris"))
    logger.setLevel(getattr(logging, get_config("LOGGER_LEVEL", "INFO" ).upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)

    log_format = get_config("LOGGER_FORMAT", LogFormat.PRETTY.value).lower()

    if log_format == LogFormat.PRETTY.value:
        handler.setFormatter(PrettyColoredFormatter())
    elif log_format == LogFormat.JSON.value:
        handler.setFormatter(JsonFormatter())
    else:
        raise ValueError(f"Unknown log format: {log_format}")

    logger.handlers = [handler]
    logger.propagate = False
    return logger


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

