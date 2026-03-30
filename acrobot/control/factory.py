"""
Controller factory for LQR design.

Separates LQR gain computation from controller runtime,
following the Factory pattern for clean initialization.
"""

import numpy as np

from ..core.config import PhysicalParams, ControlParams
from ..linearization.state_space import get_state_space
from ..linearization.controllability import is_controllable
from .lqr import design_lqr


class LQRDesignResult:
    """Immutable container for LQR design outputs."""

    __slots__ = ("_K", "_P", "_eigvals", "_A", "_B")

    def __init__(self, K: np.ndarray, P: np.ndarray,
                 eigvals: np.ndarray, A: np.ndarray, B: np.ndarray):
        self._K = K
        self._P = P
        self._eigvals = eigvals
        self._A = A
        self._B = B

    @property
    def K(self) -> np.ndarray:
        return self._K

    @property
    def P(self) -> np.ndarray:
        return self._P

    @property
    def eigenvalues(self) -> np.ndarray:
        return self._eigvals

    @property
    def A(self) -> np.ndarray:
        return self._A

    @property
    def B(self) -> np.ndarray:
        return self._B


def create_lqr(physical: PhysicalParams, ctrl: ControlParams) -> LQRDesignResult:
    """Factory function: design LQR and return all design artifacts.

    Returns:
        LQRDesignResult with K, P, eigenvalues, A, B matrices.

    Raises:
        RuntimeError: If system is not controllable.
    """
    A, B = get_state_space(physical)

    if not is_controllable(A, B):
        raise RuntimeError("System not controllable at upright equilibrium")

    K, P, eigvals = design_lqr(physical, ctrl)

    return LQRDesignResult(K=K, P=P, eigvals=eigvals, A=A, B=B)
