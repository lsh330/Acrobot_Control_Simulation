"""
Angle wrapping utilities.

Provides shared angle normalization functions used across
control, switching, and visualization modules.
Eliminates code duplication of atan2-based wrapping.
"""

import math

import numpy as np
import numba as nb


@nb.njit(cache=True, fastmath=True)
def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi] range.

    Uses atan2 for robust wrapping without branch conditions.
    """
    return math.atan2(math.sin(angle), math.cos(angle))


@nb.njit(cache=True, fastmath=True)
def angle_error_scalar(current: float, target: float) -> float:
    """Compute wrapped angle error (current - target) in [-pi, pi]."""
    return math.atan2(math.sin(current - target), math.cos(current - target))


def state_error_from_upright(state: np.ndarray, buf: np.ndarray) -> np.ndarray:
    """Compute angle-wrapped state error from upright equilibrium [pi,0,0,0].

    Writes result into pre-allocated buffer to avoid allocation.

    Args:
        state: Current state (4,)
        buf: Pre-allocated error buffer (4,), modified in-place

    Returns:
        buf (same reference, modified in-place)
    """
    buf[0] = math.atan2(math.sin(state[0] - math.pi), math.cos(state[0] - math.pi))
    buf[1] = math.atan2(math.sin(state[1]), math.cos(state[1]))
    buf[2] = state[2]
    buf[3] = state[3]
    return buf
