"""
Inertia (mass) matrix M(q) of the Acrobot.

The mass matrix is symmetric and positive-definite for all configurations.
M(q) = [[d11, d12], [d12, d22]]

where:
    d11 = alpha + 2*beta*cos(theta2)
    d12 = delta + beta*cos(theta2)
    d22 = delta

Lumped parameters (from derived.py):
    alpha = Ic1 + Ic2 + m1*lc1^2 + m2*(l1^2 + lc2^2)
    beta  = m2*l1*lc2
    delta = Ic2 + m2*lc2^2

Reference:
    Spong, M.W. (1995). Eq. (3), IEEE CSM 15(1).
"""

import math

import numba as nb


@nb.njit(cache=True, fastmath=True)
def mass_matrix_scalars(
    theta2: float,
    alpha: float,
    beta: float,
    delta: float,
) -> tuple[float, float, float]:
    """Compute mass matrix elements as scalars (zero-allocation).

    Returns:
        (d11, d12, d22) — the three unique elements of the 2x2 symmetric M(q).
    """
    c2 = math.cos(theta2)
    d11 = alpha + 2.0 * beta * c2
    d12 = delta + beta * c2
    d22 = delta
    return d11, d12, d22


@nb.njit(cache=True, fastmath=True)
def mass_matrix_det(d11: float, d12: float, d22: float) -> float:
    """Determinant of M(q): det = d11*d22 - d12^2.

    Must be strictly positive for all configurations (positive-definiteness).
    """
    return d11 * d22 - d12 * d12


@nb.njit(cache=True, fastmath=True)
def mass_matrix_inv_scalars(
    d11: float, d12: float, d22: float,
) -> tuple[float, float, float]:
    """Inverse of M(q) as scalars: M^{-1} = (1/det)*[[d22, -d12], [-d12, d11]].

    Returns:
        (m_inv_11, m_inv_12, m_inv_22) — elements of M^{-1}.
    """
    det = d11 * d22 - d12 * d12
    inv_det = 1.0 / det
    return d22 * inv_det, -d12 * inv_det, d11 * inv_det
