"""
Derived parameters computed from physical properties.

These lumped parameters simplify the equations of motion and reduce
redundant computation in the dynamics hot path.

Reference:
    Spong, M.W. (1995). Eq. (2)-(5) for the Acrobot dynamics.
"""

from dataclasses import dataclass

from ..core.config import PhysicalParams


@dataclass(frozen=True, slots=True)
class DerivedParams:
    """Precomputed lumped parameters for efficient dynamics evaluation.

    alpha = Ic1 + Ic2 + m1*lc1^2 + m2*(l1^2 + lc2^2)
    beta  = m2*l1*lc2
    delta = Ic2 + m2*lc2^2
    phi1  = (m1*lc1 + m2*l1)*g
    phi2  = m2*lc2*g
    """
    alpha: float   # Combined inertia term (constant part of d11)
    beta: float    # Coupling inertia term (coefficient of cos(theta2))
    delta: float   # Link 2 effective inertia (d22, constant)
    phi1: float    # Gravity torque coefficient for link 1
    phi2: float    # Gravity torque coefficient for link 2


def compute_derived(p: PhysicalParams) -> DerivedParams:
    """Compute derived parameters from physical properties.

    These lumped constants avoid redundant multiplications per timestep:
    - alpha, beta, delta appear in the mass matrix M(q)
    - phi1, phi2 appear in the gravity vector G(q)
    """
    alpha = p.Ic1 + p.Ic2 + p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2)
    beta = p.m2 * p.l1 * p.lc2
    delta = p.Ic2 + p.m2 * p.lc2**2
    phi1 = (p.m1 * p.lc1 + p.m2 * p.l1) * p.g
    phi2 = p.m2 * p.lc2 * p.g

    return DerivedParams(
        alpha=alpha, beta=beta, delta=delta,
        phi1=phi1, phi2=phi2,
    )


def unpack_derived_scalars(d: DerivedParams) -> tuple[float, ...]:
    """Unpack derived parameters for Numba JIT functions.

    Returns: (alpha, beta, delta, phi1, phi2)
    """
    return (d.alpha, d.beta, d.delta, d.phi1, d.phi2)
