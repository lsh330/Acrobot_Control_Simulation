"""
Equilibrium point computation for the Acrobot.

The Acrobot has four equilibrium points (zero velocity, zero torque):
    1. Downward:  [0, 0, 0, 0]      (stable)
    2. Upright:   [pi, 0, 0, 0]     (unstable — control target)
    3. Mixed 1:   [0, pi, 0, 0]     (unstable)
    4. Mixed 2:   [pi, pi, 0, 0]    (unstable)

Reference:
    Tedrake, R. Underactuated Robotics, Ch. 3.
"""

import math

import numpy as np


# Canonical equilibrium states
DOWNWARD = np.array([0.0, 0.0, 0.0, 0.0])
UPRIGHT = np.array([math.pi, 0.0, 0.0, 0.0])
MIXED_1 = np.array([0.0, math.pi, 0.0, 0.0])
MIXED_2 = np.array([math.pi, math.pi, 0.0, 0.0])


def get_upright_equilibrium() -> np.ndarray:
    """Return the upright equilibrium state [pi, 0, 0, 0]."""
    return UPRIGHT.copy()


def state_error(state: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute state error with angle wrapping.

    Angles are wrapped to [-pi, pi] for proper distance computation.
    """
    err = state - target
    # Wrap angles to [-pi, pi]
    err[0] = math.atan2(math.sin(err[0]), math.cos(err[0]))
    err[1] = math.atan2(math.sin(err[1]), math.cos(err[1]))
    return err
