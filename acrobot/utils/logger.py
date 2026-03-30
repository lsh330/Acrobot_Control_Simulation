"""
Minimal logging utility.

Avoids heavy logging frameworks; uses simple print-based output
with configurable verbosity level.
"""

import sys

_VERBOSITY: int = 1  # 0=silent, 1=normal, 2=verbose


def set_verbosity(level: int) -> None:
    """Set global verbosity level."""
    global _VERBOSITY
    _VERBOSITY = level


def info(msg: str) -> None:
    """Print info message (verbosity >= 1)."""
    if _VERBOSITY >= 1:
        print(f"[INFO] {msg}", file=sys.stderr)


def debug(msg: str) -> None:
    """Print debug message (verbosity >= 2)."""
    if _VERBOSITY >= 2:
        print(f"[DEBUG] {msg}", file=sys.stderr)


def warn(msg: str) -> None:
    """Print warning message (always shown)."""
    print(f"[WARN] {msg}", file=sys.stderr)
