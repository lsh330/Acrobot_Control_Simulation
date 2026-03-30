"""
Gravity vector G(q) of the Acrobot.

Derived from the potential energy:
    V(q) = -phi1*cos(theta1) - phi2*cos(theta1 + theta2)

    G(q) = dV/dq = [phi1*sin(theta1) + phi2*sin(theta1 + theta2),
                     phi2*sin(theta1 + theta2)]

Convention: theta1 measured from vertical downward (counterclockwise positive).
At the downward equilibrium (theta1=0, theta2=0), G=[0,0] (no restoring torque).
At the upright position (theta1=pi, theta2=0), G is also zero (unstable equilibrium).

Lumped parameters:
    phi1 = (m1*lc1 + m2*l1)*g
    phi2 = m2*lc2*g

Reference:
    Spong, M.W. (1995). Eq. (5), IEEE CSM 15(1).
"""

import math

import numba as nb


@nb.njit(cache=True, fastmath=True)
def gravity_scalars(
    theta1: float,
    theta2: float,
    phi1: float,
    phi2: float,
) -> tuple[float, float]:
    """Compute gravity torque vector as scalars (zero-allocation).

    Returns:
        (g1, g2) — the two components of G(q).
    """
    s1 = math.sin(theta1)
    s12 = math.sin(theta1 + theta2)
    g1 = phi1 * s1 + phi2 * s12
    g2 = phi2 * s12
    return g1, g2
