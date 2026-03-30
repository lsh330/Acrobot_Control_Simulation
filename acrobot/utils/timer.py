"""
Performance measurement utility.

Provides a lightweight context manager for timing code sections.
"""

import time
from contextlib import contextmanager


@contextmanager
def timer(label: str = ""):
    """Context manager that prints elapsed wall-clock time.

    Usage:
        with timer("Simulation"):
            run_simulation()
        # prints: [Simulation] 0.0123s
    """
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    if label:
        print(f"[{label}] {elapsed:.4f}s")
    else:
        print(f"{elapsed:.4f}s")
