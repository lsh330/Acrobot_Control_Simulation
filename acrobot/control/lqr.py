"""
Linear-Quadratic Regulator (LQR) for the Acrobot.

Solves the Continuous Algebraic Riccati Equation (CARE):
    A^T*P + P*A - P*B*R^{-1}*B^T*P + Q = 0

The optimal feedback gain:
    K = R^{-1}*B^T*P

Closed-loop system: dx/dt = (A - B*K)*x

All closed-loop eigenvalues must have negative real parts.

Reference:
    Anderson, B.D.O. & Moore, J.B. (1990). Optimal Control.
    Spong, M.W. (1995). IEEE CSM 15(1), Section IV.
"""

import numpy as np
from scipy.linalg import solve_continuous_are

from ..core.config import PhysicalParams, ControlParams
from ..linearization.state_space import get_state_space
from ..linearization.controllability import is_controllable


class LQRDesignError(RuntimeError):
    """Raised when LQR design fails."""
    pass


def design_lqr(
    p: PhysicalParams,
    ctrl: ControlParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design the LQR controller for the Acrobot.

    Args:
        p: Physical parameters
        ctrl: Control parameters (Q_diag, R)

    Returns:
        (K, P, eigvals) where:
            K: (1, 4) feedback gain matrix
            P: (4, 4) solution to CARE
            eigvals: (4,) closed-loop eigenvalues

    Raises:
        LQRDesignError: If system is not controllable or CARE fails.
    """
    A, B = get_state_space(p)

    if not is_controllable(A, B):
        raise LQRDesignError("System is not controllable at the upright equilibrium.")

    Q = np.diag(ctrl.Q_diag)
    R = np.array([[ctrl.R]])

    # Solve CARE: A^T*P + P*A - P*B*R^{-1}*B^T*P + Q = 0
    P = solve_continuous_are(A, B, Q, R)

    # Feedback gain: K = R^{-1}*B^T*P
    K = np.linalg.solve(R, B.T @ P)  # (1, 4)

    # Closed-loop eigenvalues for verification
    A_cl = A - B @ K
    eigvals = np.linalg.eigvals(A_cl)

    # Verify all eigenvalues have negative real part
    if np.any(eigvals.real >= 0):
        raise LQRDesignError(
            f"LQR design failed: unstable closed-loop poles {eigvals}")

    return K, P, eigvals


def lqr_control(
    state: np.ndarray,
    target: np.ndarray,
    K: np.ndarray,
    max_torque: float,
) -> float:
    """Compute LQR control input u = -K*(x - x_target).

    Includes torque saturation to respect actuator limits.

    Args:
        state: Current state (4,)
        target: Target equilibrium (4,)
        K: LQR gain matrix (1, 4)
        max_torque: Maximum allowed torque [N*m]

    Returns:
        Saturated control torque (scalar).
    """
    from ..utils.angle import wrap_angle
    err = state - target
    err[0] = wrap_angle(err[0])
    err[1] = wrap_angle(err[1])

    u = float((-K @ err).item())

    # Saturation
    if u > max_torque:
        return max_torque
    if u < -max_torque:
        return -max_torque
    return u
