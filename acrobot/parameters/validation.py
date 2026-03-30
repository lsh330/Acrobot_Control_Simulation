"""
Parameter validation ensuring physical consistency.

Validates that all physical parameters satisfy fundamental constraints:
- Positive masses, lengths, and inertias
- Center of mass within link length
- Parallel axis theorem consistency
"""

from ..core.config import PhysicalParams


class ParameterValidationError(ValueError):
    """Raised when physical parameters violate constraints."""
    pass


def validate_params(p: PhysicalParams) -> None:
    """Validate physical parameter consistency.

    Checks:
        1. All masses, lengths, inertias are strictly positive
        2. Center of mass within link length: 0 < lci <= li
        3. Parallel axis theorem lower bound: Ici >= 0
           (Ici is about CoM, so any positive value is valid)
        4. Damping coefficients are non-negative
        5. Gravity is positive

    Raises:
        ParameterValidationError: If any constraint is violated.
    """
    # Positive mass
    if p.m1 <= 0 or p.m2 <= 0:
        raise ParameterValidationError(
            f"Masses must be positive: m1={p.m1}, m2={p.m2}")

    # Positive lengths
    if p.l1 <= 0 or p.l2 <= 0:
        raise ParameterValidationError(
            f"Link lengths must be positive: l1={p.l1}, l2={p.l2}")

    # Center of mass within link
    if not (0 < p.lc1 <= p.l1):
        raise ParameterValidationError(
            f"CoM must be within link: 0 < lc1={p.lc1} <= l1={p.l1}")
    if not (0 < p.lc2 <= p.l2):
        raise ParameterValidationError(
            f"CoM must be within link: 0 < lc2={p.lc2} <= l2={p.l2}")

    # Positive inertias
    if p.Ic1 <= 0 or p.Ic2 <= 0:
        raise ParameterValidationError(
            f"Inertias must be positive: Ic1={p.Ic1}, Ic2={p.Ic2}")

    # Non-negative damping
    if p.b1 < 0 or p.b2 < 0:
        raise ParameterValidationError(
            f"Damping must be non-negative: b1={p.b1}, b2={p.b2}")

    # Positive gravity
    if p.g <= 0:
        raise ParameterValidationError(
            f"Gravity must be positive: g={p.g}")

    # Parallel axis theorem sanity: I_total = Ic + m*lc^2
    # The total inertia about the pivot should be reasonable
    I1_pivot = p.Ic1 + p.m1 * p.lc1**2
    I2_pivot = p.Ic2 + p.m2 * p.lc2**2
    if I1_pivot <= 0 or I2_pivot <= 0:
        raise ParameterValidationError(
            f"Pivot inertias must be positive: I1={I1_pivot}, I2={I2_pivot}")
