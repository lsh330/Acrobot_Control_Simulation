"""
Energy computations for the Acrobot.

Kinetic energy:  T = (1/2)*q_dot^T * M(q) * q_dot
Potential energy: V = -phi1*cos(theta1) - phi2*cos(theta1 + theta2)
Total energy:     E = T + V

The upright equilibrium energy (maximum potential):
    E_upright = phi1 + phi2

Reference:
    Xin, X. & Kaneda, M. (2007). Eq. (3)-(5), IJRNC 17(16).
"""

import math

import numba as nb

from .mass_matrix import mass_matrix_scalars


@nb.njit(cache=True, fastmath=True)
def kinetic_energy(
    theta2: float,
    dtheta1: float,
    dtheta2: float,
    alpha: float,
    beta: float,
    delta: float,
) -> float:
    """Compute kinetic energy T = (1/2)*dq^T*M(q)*dq.

    Expanded: T = (1/2)*(d11*dq1^2 + 2*d12*dq1*dq2 + d22*dq2^2)
    """
    d11, d12, d22 = mass_matrix_scalars(theta2, alpha, beta, delta)
    return 0.5 * (d11 * dtheta1 * dtheta1
                  + 2.0 * d12 * dtheta1 * dtheta2
                  + d22 * dtheta2 * dtheta2)


@nb.njit(cache=True, fastmath=True)
def potential_energy(
    theta1: float,
    theta2: float,
    phi1: float,
    phi2: float,
) -> float:
    """Compute potential energy V = -phi1*cos(q1) - phi2*cos(q1+q2).

    Reference datum: V=0 at the downward equilibrium (theta1=0, theta2=0).
    At upright (theta1=pi, theta2=0): V = phi1 + phi2 (maximum).
    """
    return -phi1 * math.cos(theta1) - phi2 * math.cos(theta1 + theta2)


@nb.njit(cache=True, fastmath=True)
def total_energy(
    theta1: float,
    theta2: float,
    dtheta1: float,
    dtheta2: float,
    alpha: float,
    beta: float,
    delta: float,
    phi1: float,
    phi2: float,
) -> float:
    """Compute total mechanical energy E = T + V."""
    T = kinetic_energy(theta2, dtheta1, dtheta2, alpha, beta, delta)
    V = potential_energy(theta1, theta2, phi1, phi2)
    return T + V


@nb.njit(cache=True, fastmath=True)
def upright_energy(phi1: float, phi2: float) -> float:
    """Energy at the upright equilibrium (theta1=pi, theta2=0, zero velocity).

    E_upright = phi1 + phi2  (maximum potential energy, zero kinetic)
    """
    return phi1 + phi2
