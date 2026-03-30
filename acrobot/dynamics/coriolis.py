"""
Coriolis and centrifugal force matrix C(q, dq) of the Acrobot.

Derived via Christoffel symbols of the first kind:
    c_{ijk} = (1/2)*(dM_{ij}/dq_k + dM_{ik}/dq_j - dM_{jk}/dq_i)

For the Acrobot, only dM/dtheta2 is non-zero:
    dM_{11}/dtheta2 = -2*beta*sin(theta2)
    dM_{12}/dtheta2 = -beta*sin(theta2)
    dM_{22}/dtheta2 = 0

This yields:
    h = -beta*sin(theta2)
    C = [[h*dtheta2,          h*(dtheta1 + dtheta2)],
         [-h*dtheta1,         0                    ]]

Property: N(q,dq) = dM/dt - 2*C is skew-symmetric (passivity).

Reference:
    Spong, M.W. (1995). Eq. (4), IEEE CSM 15(1).
    Murray, Li & Sastry (1994). A Mathematical Introduction to Robotic
    Manipulation, Ch. 4.
"""

import math

import numba as nb


@nb.njit(cache=True, fastmath=True)
def coriolis_scalars(
    theta2: float,
    dtheta1: float,
    dtheta2: float,
    beta: float,
) -> tuple[float, float, float, float]:
    """Compute Coriolis matrix elements as scalars (zero-allocation).

    Returns:
        (c11, c12, c21, c22) — the four elements of C(q, dq).
    """
    h = -beta * math.sin(theta2)
    c11 = h * dtheta2
    c12 = h * (dtheta1 + dtheta2)
    c21 = -h * dtheta1
    c22 = 0.0
    return c11, c12, c21, c22


@nb.njit(cache=True, fastmath=True)
def coriolis_torques(
    theta2: float,
    dtheta1: float,
    dtheta2: float,
    beta: float,
) -> tuple[float, float]:
    """Compute C(q,dq)*dq as a 2-vector (the actual torque contribution).

    Returns:
        (tau_c1, tau_c2) where tau_c = C(q,dq) * [dtheta1, dtheta2]^T.
    """
    c11, c12, c21, c22 = coriolis_scalars(theta2, dtheta1, dtheta2, beta)
    tau_c1 = c11 * dtheta1 + c12 * dtheta2
    tau_c2 = c21 * dtheta1 + c22 * dtheta2
    return tau_c1, tau_c2
