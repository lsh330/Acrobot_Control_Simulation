"""
Hybrid controller: energy-based swing-up + LQR balancing.

Implements ControllerBase interface. Uses LQR factory for
clean separation of design-time and runtime concerns.

Reference:
    Spong, M.W. (1995). IEEE CSM 15(1), 49-55.
"""

import math

import numpy as np

from ..core.config import PhysicalParams, ControlParams
from ..parameters.derived import compute_derived
from .base import ControllerBase
from .factory import create_lqr
from .lqr import lqr_control
from .energy_swing_up import energy_swing_up_control
from .switching import SwitchingController


class HybridController(ControllerBase):
    """Energy swing-up + LQR hybrid controller.

    Implements ControllerBase for interchangeability.
    Uses pre-computed derived parameters and LQR design
    from the factory to minimize runtime overhead.
    """

    __slots__ = (
        "_lqr_result", "_switcher", "_derived",
        "_x_eq", "_ctrl", "_k_p", "_k_d",
    )

    def __init__(self, physical: PhysicalParams, ctrl: ControlParams) -> None:
        self._ctrl = ctrl
        self._derived = compute_derived(physical)
        self._x_eq = np.array([math.pi, 0.0, 0.0, 0.0])

        # Pre-compute PFL gains (eliminates 120k+ multiplications per sim)
        self._k_p = 10.0 * ctrl.k_energy
        self._k_d = 1.0 * ctrl.k_energy

        # LQR design via factory
        self._lqr_result = create_lqr(physical, ctrl)

        # Switching logic with Lyapunov metric
        self._switcher = SwitchingController(
            P=self._lqr_result.P,
            switching_radius=ctrl.switching_radius,
            hysteresis=ctrl.hysteresis,
        )

    def compute_control(self, state: np.ndarray) -> float:
        """Compute control torque (ControllerBase interface)."""
        if self._switcher.should_use_lqr(state):
            return lqr_control(
                state, self._x_eq, self._lqr_result.K, self._ctrl.max_torque)
        else:
            d = self._derived
            return energy_swing_up_control(
                state[0], state[1], state[2], state[3],
                self._ctrl.k_energy, self._ctrl.max_torque,
                d.alpha, d.beta, d.delta, d.phi1, d.phi2,
            )

    def reset(self) -> None:
        """Reset controller to swing-up mode."""
        self._switcher.reset()

    @property
    def mode_name(self) -> str:
        return "LQR" if self._switcher.is_using_lqr else "Swing-Up"

    @property
    def K(self) -> np.ndarray:
        return self._lqr_result.K

    @property
    def P(self) -> np.ndarray:
        return self._lqr_result.P

    @property
    def closed_loop_eigenvalues(self) -> np.ndarray:
        return self._lqr_result.eigenvalues

    @property
    def is_using_lqr(self) -> bool:
        return self._switcher.is_using_lqr
