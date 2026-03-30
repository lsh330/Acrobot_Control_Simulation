"""
Initial condition generators for the Acrobot simulation.

Provides standard initial states and random perturbation utilities
for Monte Carlo analysis.
"""

import math

import numpy as np


def downward_rest() -> np.ndarray:
    """Standard initial condition: hanging downward, zero velocity."""
    return np.array([0.0, 0.0, 0.0, 0.0])


def from_tuple(state: tuple[float, ...]) -> np.ndarray:
    """Create initial state from a tuple."""
    return np.array(state, dtype=np.float64)


def random_near_upright(
    rng: np.random.Generator,
    angle_std: float = 0.1,
    vel_std: float = 0.1,
) -> np.ndarray:
    """Random initial condition near the upright equilibrium.

    Useful for testing LQR performance and ROA estimation.
    """
    return np.array([
        math.pi + rng.normal(0, angle_std),
        rng.normal(0, angle_std),
        rng.normal(0, vel_std),
        rng.normal(0, vel_std),
    ])


def random_state(
    rng: np.random.Generator,
    angle_range: float = math.pi,
    vel_range: float = 2.0,
) -> np.ndarray:
    """Uniformly random state for Monte Carlo analysis."""
    return np.array([
        rng.uniform(-angle_range, angle_range),
        rng.uniform(-angle_range, angle_range),
        rng.uniform(-vel_range, vel_range),
        rng.uniform(-vel_range, vel_range),
    ])
