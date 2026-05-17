"""Logger factory with a sensible default handler.

`logging.getLogger` returns a global singleton, so re-configuring it from each
script entry-point would stack handlers on repeated imports. The helper here
installs a single stream handler the first time it is called and reuses it.
"""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger named ``name`` with a stream handler attached.

    Args:
        name: Module-style dotted name, conventionally ``__name__``.
        level: Logging level applied to both the logger and its handler.

    Returns:
        A logger that writes to ``sys.stderr`` with a project-wide format.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Without this guard, importing a module twice in a notebook stacks handlers
    # and each log line is emitted N times.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATEFMT))
        logger.addHandler(handler)
        logger.propagate = False

    return logger
