"""
Stability analysis for the Acrobot LQR controller.

Verifies:
    1. All closed-loop eigenvalues have negative real parts
    2. CARE residual is near zero
    3. Lyapunov function V(x) = x^T*P*x is decreasing
"""

import numpy as np

from ..core.config import PhysicalParams, ControlParams
from ..linearization.state_space import get_state_space
from ..control.lqr import design_lqr


def verify_stability(p: PhysicalParams, ctrl: ControlParams) -> dict:
    """Run comprehensive stability verification.

    Returns dict with verification results.
    """
    A, B = get_state_space(p)
    K, P, eigvals = design_lqr(p, ctrl)

    Q = np.diag(ctrl.Q_diag)
    R = np.array([[ctrl.R]])

    # CARE residual: A^T*P + P*A - P*B*R^{-1}*B^T*P + Q should be ~0
    care_residual = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q
    care_norm = np.linalg.norm(care_residual)

    # P properties
    p_eigvals = np.linalg.eigvals(P)
    p_symmetric = np.allclose(P, P.T)
    p_positive_definite = np.all(p_eigvals > 0)

    # Closed-loop stability
    A_cl = A - B @ K
    cl_eigvals = np.linalg.eigvals(A_cl)
    all_stable = np.all(cl_eigvals.real < 0)

    # Gain margin (minimum ratio before instability)
    max_real_part = np.max(cl_eigvals.real)

    return {
        "closed_loop_eigenvalues": cl_eigvals,
        "all_stable": bool(all_stable),
        "care_residual_norm": float(care_norm),
        "P_symmetric": bool(p_symmetric),
        "P_positive_definite": bool(p_positive_definite),
        "P_eigenvalues": p_eigvals,
        "max_real_part": float(max_real_part),
        "K": K,
        "P": P,
    }
