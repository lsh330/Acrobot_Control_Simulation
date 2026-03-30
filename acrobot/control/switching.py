"""
Controller switching logic for hybrid control.

Manages the transition from swing-up to LQR balancing
with hysteresis to prevent chattering at the boundary.

Uses pre-allocated error buffer to avoid heap allocation
in the hot loop (60k+ calls per simulation).

Reference:
    Spong, M.W. (1995). Section V, IEEE CSM 15(1).
"""

import numpy as np

from ..utils.angle import state_error_from_upright


class SwitchingController:
    """Manages swing-up to LQR switching with hysteresis.

    Uses pre-allocated buffer for state error computation
    to eliminate per-call numpy array allocation.
    """

    __slots__ = ("_rho", "_hysteresis", "_P", "_using_lqr", "_err_buf")

    def __init__(
        self,
        P: np.ndarray,
        switching_radius: float,
        hysteresis: float,
    ) -> None:
        self._P = P
        self._rho = switching_radius
        self._hysteresis = hysteresis
        self._using_lqr = False
        self._err_buf = np.empty(4, dtype=np.float64)  # pre-allocated

    def lyapunov_distance(self, state: np.ndarray) -> float:
        """Compute weighted distance using pre-allocated buffer."""
        err = state_error_from_upright(state, self._err_buf)
        return float(err @ self._P @ err)

    def should_use_lqr(self, state: np.ndarray) -> bool:
        """Determine controller mode with Lyapunov-based hysteresis."""
        dist = self.lyapunov_distance(state)

        if self._using_lqr:
            if dist > self._rho * (1.0 + self._hysteresis):
                self._using_lqr = False
        else:
            if dist < self._rho:
                self._using_lqr = True

        return self._using_lqr

    def reset(self) -> None:
        """Reset to swing-up mode."""
        self._using_lqr = False

    @property
    def is_using_lqr(self) -> bool:
        return self._using_lqr
