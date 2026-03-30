"""
Controllability analysis for the linearized Acrobot.

Verifies the Kalman controllability condition:
    rank([B, AB, A^2B, A^3B]) = n (full rank)

This is a necessary condition for LQR design.

Reference:
    Kalman, R.E. (1960). On the general theory of control systems.
    Anderson, B.D.O. & Moore, J.B. (1990). Optimal Control, Ch. 4.
"""

import numpy as np


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the controllability matrix [B, AB, ..., A^{n-1}B].

    Args:
        A: (n, n) state matrix
        B: (n, m) input matrix

    Returns:
        (n, n*m) controllability matrix
    """
    n = A.shape[0]
    cols = [B]
    Ab = B.copy()
    for _ in range(n - 1):
        Ab = A @ Ab
        cols.append(Ab)
    return np.hstack(cols)


def is_controllable(A: np.ndarray, B: np.ndarray) -> bool:
    """Check if the system (A, B) is controllable.

    Returns True if rank of controllability matrix equals state dimension.
    """
    C_mat = controllability_matrix(A, B)
    return int(np.linalg.matrix_rank(C_mat)) == A.shape[0]


def controllability_rank(A: np.ndarray, B: np.ndarray) -> int:
    """Return the rank of the controllability matrix."""
    C_mat = controllability_matrix(A, B)
    return int(np.linalg.matrix_rank(C_mat))
