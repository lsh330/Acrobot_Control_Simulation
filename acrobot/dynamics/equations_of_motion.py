"""
Equations of Motion (EOM) for the Acrobot.

The EOM in standard manipulator form:
    M(q) * q_ddot + C(q, q_dot) * q_dot + G(q) = B * u

where B = [0, 1]^T (only the elbow joint is actuated).

Rearranging for q_ddot:
    q_ddot = M(q)^{-1} * (B*u - C(q,q_dot)*q_dot - G(q))

Including viscous damping:
    q_ddot = M(q)^{-1} * (B*u - C(q,q_dot)*q_dot - G(q) - D*q_dot)

where D = diag(b1, b2) is the damping matrix.

This module provides the state derivative function f(x, u) for numerical
integration: dx/dt = f(x, u).

Reference:
    Spong, M.W. (1995). Eq. (1)-(5), IEEE CSM 15(1).
"""

import math

import numba as nb

from .mass_matrix import mass_matrix_scalars, mass_matrix_inv_scalars
from .coriolis import coriolis_torques
from .gravity import gravity_scalars


@nb.njit(cache=True, fastmath=True)
def state_derivative(
    theta1: float,
    theta2: float,
    dtheta1: float,
    dtheta2: float,
    u: float,
    alpha: float,
    beta: float,
    delta: float,
    phi1: float,
    phi2: float,
    b1: float,
    b2: float,
) -> tuple[float, float, float, float]:
    """Compute dx/dt = f(x, u) for the Acrobot.

    State: x = [theta1, theta2, dtheta1, dtheta2]
    Input: u = torque at elbow joint

    Returns:
        (dtheta1, dtheta2, ddtheta1, ddtheta2)

    All computation uses scalar arithmetic (zero heap allocation).
    """
    # Mass matrix elements
    d11, d12, d22 = mass_matrix_scalars(theta2, alpha, beta, delta)

    # Coriolis torques: C(q,dq)*dq
    tau_c1, tau_c2 = coriolis_torques(theta2, dtheta1, dtheta2, beta)

    # Gravity torques: G(q)
    g1, g2 = gravity_scalars(theta1, theta2, phi1, phi2)

    # Right-hand side: B*u - C*dq - G - D*dq
    # B = [0, 1]^T, so u only enters the second equation
    rhs1 = -tau_c1 - g1 - b1 * dtheta1
    rhs2 = u - tau_c2 - g2 - b2 * dtheta2

    # Solve M * q_ddot = rhs via explicit 2x2 inverse
    m_inv_11, m_inv_12, m_inv_22 = mass_matrix_inv_scalars(d11, d12, d22)

    ddtheta1 = m_inv_11 * rhs1 + m_inv_12 * rhs2
    ddtheta2 = m_inv_12 * rhs1 + m_inv_22 * rhs2

    return dtheta1, dtheta2, ddtheta1, ddtheta2
