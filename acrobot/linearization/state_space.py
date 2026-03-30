"""
State-space model (A, B) for the linearized Acrobot.

Delegates to the analytical Jacobian in dynamics/jacobian.py.
This module provides a clean interface for the control design modules.
"""

import numpy as np

from ..core.config import PhysicalParams
from ..parameters.derived import compute_derived
from ..dynamics.jacobian import linearize_at_upright_matrices


def get_state_space(p: PhysicalParams) -> tuple[np.ndarray, np.ndarray]:
    """Compute the linearized state-space matrices at the upright equilibrium.

    Returns:
        (A, B) where A is (4,4) and B is (4,1).
    """
    d = compute_derived(p)
    return linearize_at_upright_matrices(
        d.alpha, d.beta, d.delta, d.phi1, d.phi2, p.b1, p.b2)
