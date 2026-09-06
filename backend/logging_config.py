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

# All app loggers live under this namespace ("docmind.<module>") so one config
# controls them and library logging is left untouched.
ROOT_LOGGER_NAME = "docmind"

# Format includes a timestamp and the level, e.g.:
#   2026-09-05 18:20:01 [INFO] docmind.retriever: Loading documents
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

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

    # Idempotency guard: only attach a handler the first time.
    if not getattr(logger, "_docmind_configured", False):
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(handler)
        # Don't also bubble up to the root logger's handlers (avoids double lines
        # if the host application configured the root logger too).
        logger.propagate = False
        logger._docmind_configured = True  # type: ignore[attr-defined]

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
