"""Centralized logging configuration for DocMind AI.

One place configures logging for the whole app; every other module just calls
``get_logger(__name__)`` and logs. This keeps handler setup out of the
individual pipeline modules and guarantees a single, consistent format.

Design notes
------------
* Output goes to ``stdout`` so the lines show up in the terminal running
  ``streamlit run app.py`` (Streamlit surfaces the server process's stdout).
* ``configure_logging`` is **idempotent**. Streamlit re-executes ``app.py`` on
  every interaction, and importing modules repeatedly must not stack duplicate
  handlers (which would print each line many times). A sentinel flag on the
  DocMind root logger makes repeat calls a no-op.
* We configure only the ``docmind`` logger namespace, not the root logger, so
  we do not disturb or duplicate the logging of Streamlit, urllib3, httpx,
  sentence-transformers, etc.
* Standard library only -- no third-party logging dependency.
"""

from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar
from pathlib import Path
from logging.handlers import RotatingFileHandler
import re

# All app loggers live under this namespace ("docmind.<module>") so one config
# controls them and library logging is left untouched.
ROOT_LOGGER_NAME = "docmind"

# One format is used by both console and rotating-file handlers.
LOG_FORMAT = "[%(asctime)s,%(msecs)03d] [%(levelname)s] [Session: %(session_id)s] [%(component)s] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "docmind.log"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3
_CONFIG_VERSION = 2

_session_id: ContextVar[str] = ContextVar("docmind_session_id", default="default_session")
_SAFE_VALUE_RE = re.compile(r"[^\w .:@/+\-]", re.ASCII)


def sanitize_log_value(value: object, fallback: str = "unknown", limit: int = 160) -> str:
  """Keep user-controlled values single-line and bounded in log output."""
  text = str(value).replace("\r", " ").replace("\n", " ").strip()
  text = _SAFE_VALUE_RE.sub("_", text)
  return (text or fallback)[:limit]


class _ContextFilter(logging.Filter):
  """Add request context without requiring every call site to pass it."""

  def filter(self, record: logging.LogRecord) -> bool:
    record.session_id = _session_id.get()
    if not hasattr(record, "component"):
      component_names = {
        "app": "App",
        "citations": "Citations",
        "faiss_store": "Indexer",
        "generator": "Generator",
        "hybrid_retrieval": "Retriever",
        "rag_pipeline": "RAG",
        "retriever": "Retriever",
      }
      module = record.name.rsplit(".", 1)[-1]
      record.component = component_names.get(module, module.title())
    return True


def set_session_id(session_id: str) -> None:
  """Set the existing application session ID for logs in this execution."""
  _session_id.set(sanitize_log_value(session_id, fallback="default_session", limit=80))


def get_session_id() -> str:
  return _session_id.get()

# Level can be overridden with DOCMIND_LOG_LEVEL=DEBUG (etc.) without code
# changes; defaults to INFO for normal application flow.
DEFAULT_LEVEL = os.getenv("DOCMIND_LOG_LEVEL", "INFO").upper()


def configure_logging(level: str | int | None = None) -> logging.Logger:
    """Configure the DocMind logger once and return it.

    Safe to call many times (Streamlit reruns): the handler is attached only on
    the first call. Returns the ``docmind`` root logger.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)

    resolved_level = level if level is not None else DEFAULT_LEVEL
    logger.setLevel(resolved_level)

    if getattr(logger, "_docmind_config_version", None) != _CONFIG_VERSION:
      # Remove handlers from an older configuration as well as from a first
      # configuration, preventing mixed formats after a Streamlit reload.
      for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

      formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
      context_filter = _ContextFilter()

      console_handler = logging.StreamHandler(stream=sys.stdout)
      console_handler.setFormatter(formatter)
      console_handler.addFilter(context_filter)
      logger.addHandler(console_handler)

      try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
          LOG_FILE,
          maxBytes=MAX_BYTES,
          backupCount=BACKUP_COUNT,
          encoding="utf-8",
        )
      except OSError:
        # Console logging remains available if local file logging cannot be
        # initialized; logging must never prevent the app from starting.
        file_handler = None
      if file_handler is not None:
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)
        logger.addHandler(file_handler)

      # Don't also bubble up to the root logger's handlers.
      logger.propagate = False
      logger._docmind_config_version = _CONFIG_VERSION  # type: ignore[attr-defined]

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a module logger under the DocMind namespace.

    ``get_logger(__name__)`` yields e.g. ``docmind.backend.retriever``. Logging
    is configured on first use so importing any instrumented module is enough to
    get formatted terminal output, with no per-module setup.
    """
    configure_logging()
    if not name or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)
    # Nest every module logger under the DocMind namespace.
    short = name.split(".")[-1]
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{short}")
