"""
Physical parameters of the Acrobot system.

Provides the Spong/Drake canonical parameter set and utilities
for extracting scalar values for Numba JIT hot paths.

Reference:
    Spong, M.W. (1995). IEEE Control Systems Magazine, 15(1), 49-55.
    Drake: https://drake.mit.edu/
"""

from ..core.config import PhysicalParams


def get_default_params() -> PhysicalParams:
    """Return the Spong/Drake canonical parameter set."""
    return PhysicalParams()


def unpack_scalars(p: PhysicalParams) -> tuple[float, ...]:
    """Unpack physical parameters into a flat scalar tuple.

    Returns scalars in order:
        (m1, m2, l1, l2, lc1, lc2, Ic1, Ic2, b1, b2, g)

    This format is optimal for Numba @njit functions which
    cannot accept dataclass arguments.
    """
    return (p.m1, p.m2, p.l1, p.l2, p.lc1, p.lc2,
            p.Ic1, p.Ic2, p.b1, p.b2, p.g)
