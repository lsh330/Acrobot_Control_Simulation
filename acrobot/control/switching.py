"""
Controller switching logic for hybrid control.

Manages the transition from swing-up to LQR balancing
with hysteresis to prevent chattering at the boundary.

The switching condition is based on a weighted state-space
distance to the upright equilibrium, using the Lyapunov
matrix P from the LQR design as the metric:

    d(x) = (x - x_eq)^T * P * (x - x_eq)

When d(x) < rho (ROA threshold), switch to LQR.
When d(x) > rho + hysteresis, switch back to swing-up.

Reference:
    Spong, M.W. (1995). Section V, IEEE CSM 15(1).
"""

import math

import numpy as np


class SwitchingController:
    """Manages swing-up to LQR switching with hysteresis."""

    __slots__ = ("_rho", "_hysteresis", "_P", "_x_eq", "_using_lqr")

    def __init__(
        self,
        P: np.ndarray,
        switching_radius: float,
        hysteresis: float,
    ) -> None:
        """Initialize the switching controller.

        Args:
            P: (4,4) Lyapunov/CARE solution matrix
            switching_radius: Threshold for LQR activation
            hysteresis: Band width to prevent chattering
        """
        self._P = P
        self._rho = switching_radius
        self._hysteresis = hysteresis
        self._x_eq = np.array([math.pi, 0.0, 0.0, 0.0])
        self._using_lqr = False

    def state_error(self, state: np.ndarray) -> np.ndarray:
        """Compute angle-wrapped state error from upright."""
        err = state - self._x_eq
        err[0] = math.atan2(math.sin(err[0]), math.cos(err[0]))
        err[1] = math.atan2(math.sin(err[1]), math.cos(err[1]))
        return err

    def lyapunov_distance(self, state: np.ndarray) -> float:
        """Compute weighted distance: (x-x_eq)^T * P * (x-x_eq)."""
        err = self.state_error(state)
        return float(err @ self._P @ err)

    def angle_distance(self, state: np.ndarray) -> float:
        """Compute simple angular distance to upright.

        d = |delta_theta1| + |delta_theta2| + 0.1*(|dtheta1| + |dtheta2|)
        """
        err = self.state_error(state)
        return abs(err[0]) + abs(err[1]) + 0.1 * (abs(err[2]) + abs(err[3]))

    def should_use_lqr(self, state: np.ndarray) -> bool:
        """Determine whether to use LQR (True) or swing-up (False).

        Uses Lyapunov distance x^T*P*x for both activation and
        deactivation. The P matrix from CARE naturally weights
        angles vs velocities appropriately for the ROA.

        Hysteresis: activate at rho, deactivate at rho*(1+hysteresis).
        """
        dist = self.lyapunov_distance(state)

        if self._using_lqr:
            if dist > self._rho * (1.0 + self._hysteresis):
                self._using_lqr = False
        else:
            if dist < self._rho:
                self._using_lqr = True

        return self._using_lqr

    @property
    def is_using_lqr(self) -> bool:
        """Current controller mode."""
        return self._using_lqr
