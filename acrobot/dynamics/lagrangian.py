"""
Lagrangian formulation of the Acrobot dynamics.

The Lagrangian L = T - V is the foundation of the EOM derivation.
This module provides the Lagrangian value for verification and
energy-based analysis.

T = (1/2)*dq^T*M(q)*dq  (kinetic energy)
V = -phi1*cos(q1) - phi2*cos(q1+q2)  (potential energy)
L = T - V

Euler-Lagrange equations: d/dt(dL/d(dqi)) - dL/dqi = tau_i

Reference:
    Goldstein, H. (2002). Classical Mechanics, 3rd ed., Ch. 2.
    Spong, M.W. (1995). IEEE CSM 15(1).
"""

import numba as nb

from .energy import kinetic_energy, potential_energy


@nb.njit(cache=True, fastmath=True)
def lagrangian(
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
    """Compute the Lagrangian L = T - V."""
    T = kinetic_energy(theta2, dtheta1, dtheta2, alpha, beta, delta)
    V = potential_energy(theta1, theta2, phi1, phi2)
    return T - V
