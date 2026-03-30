"""
Hybrid controller: energy-based swing-up + LQR balancing.

Combines the energy swing-up controller for the nonlinear regime
with LQR for local stabilization near the upright equilibrium.

The switching logic uses a Lyapunov-based distance metric with
hysteresis to prevent chattering.

Reference:
    Spong, M.W. (1995). "The Swing Up Control Problem for the Acrobot."
    IEEE Control Systems Magazine, 15(1), 49-55.
"""

import math

import numpy as np

from ..core.config import PhysicalParams, ControlParams
from ..parameters.derived import compute_derived, DerivedParams
from .lqr import design_lqr, lqr_control
from .energy_swing_up import energy_swing_up_control
from .switching import SwitchingController


class HybridController:
    """Energy swing-up + LQR hybrid controller."""

    __slots__ = (
        "_K", "_P", "_eigvals", "_switcher", "_derived",
        "_x_eq", "_ctrl", "_physical",
    )

    def __init__(self, physical: PhysicalParams, ctrl: ControlParams) -> None:
        self._physical = physical
        self._ctrl = ctrl
        self._derived = compute_derived(physical)
        self._x_eq = np.array([math.pi, 0.0, 0.0, 0.0])

        # Design LQR
        self._K, self._P, self._eigvals = design_lqr(physical, ctrl)

        # Initialize switching logic
        self._switcher = SwitchingController(
            P=self._P,
            switching_radius=ctrl.switching_radius,
            hysteresis=ctrl.hysteresis,
        )

    def compute_control(self, state: np.ndarray) -> float:
        """Compute control torque for the current state.

        Automatically selects between swing-up and LQR based
        on the switching condition.
        """
        if self._switcher.should_use_lqr(state):
            return lqr_control(
                state, self._x_eq, self._K, self._ctrl.max_torque)
        else:
            d = self._derived
            return energy_swing_up_control(
                state[0], state[1], state[2], state[3],
                self._ctrl.k_energy, self._ctrl.max_torque,
                d.alpha, d.beta, d.delta, d.phi1, d.phi2,
            )

    @property
    def K(self) -> np.ndarray:
        return self._K

    @property
    def P(self) -> np.ndarray:
        return self._P

    @property
    def closed_loop_eigenvalues(self) -> np.ndarray:
        return self._eigvals

    @property
    def is_using_lqr(self) -> bool:
        return self._switcher.is_using_lqr
