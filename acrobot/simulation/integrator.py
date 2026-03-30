"""
Numerical integrators for the Acrobot simulation.

Provides:
    1. RK4 (classical 4th-order Runge-Kutta) — fixed step, O(h^4)

All integrators operate on scalar state to avoid heap allocation
in the inner loop (Numba JIT compiled).

Reference:
    Dormand, J.R. & Prince, P.J. (1980). "A family of embedded
    Runge-Kutta formulae." J. Comp. Appl. Math., 6, 19-26.
"""

import numba as nb

from ..dynamics.equations_of_motion import state_derivative


@nb.njit(cache=True, fastmath=True)
def rk4_step(
    theta1: float, theta2: float,
    dtheta1: float, dtheta2: float,
    u: float, dt: float,
    alpha: float, beta: float, delta: float,
    phi1: float, phi2: float,
    b1: float, b2: float,
) -> tuple[float, float, float, float]:
    """Single RK4 integration step (zero-allocation).

    Each k-evaluation computes the full 4D state derivative inline.
    """
    # k1
    k1_1, k1_2, k1_3, k1_4 = state_derivative(
        theta1, theta2, dtheta1, dtheta2, u,
        alpha, beta, delta, phi1, phi2, b1, b2)

    # k2
    h2 = 0.5 * dt
    k2_1, k2_2, k2_3, k2_4 = state_derivative(
        theta1 + h2 * k1_1, theta2 + h2 * k1_2,
        dtheta1 + h2 * k1_3, dtheta2 + h2 * k1_4, u,
        alpha, beta, delta, phi1, phi2, b1, b2)

    # k3
    k3_1, k3_2, k3_3, k3_4 = state_derivative(
        theta1 + h2 * k2_1, theta2 + h2 * k2_2,
        dtheta1 + h2 * k2_3, dtheta2 + h2 * k2_4, u,
        alpha, beta, delta, phi1, phi2, b1, b2)

    # k4
    k4_1, k4_2, k4_3, k4_4 = state_derivative(
        theta1 + dt * k3_1, theta2 + dt * k3_2,
        dtheta1 + dt * k3_3, dtheta2 + dt * k3_4, u,
        alpha, beta, delta, phi1, phi2, b1, b2)

    # Weighted average
    sixth = dt / 6.0
    new_theta1 = theta1 + sixth * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1)
    new_theta2 = theta2 + sixth * (k1_2 + 2.0 * k2_2 + 2.0 * k3_2 + k4_2)
    new_dtheta1 = dtheta1 + sixth * (k1_3 + 2.0 * k2_3 + 2.0 * k3_3 + k4_3)
    new_dtheta2 = dtheta2 + sixth * (k1_4 + 2.0 * k2_4 + 2.0 * k3_4 + k4_4)

    return new_theta1, new_theta2, new_dtheta1, new_dtheta2
