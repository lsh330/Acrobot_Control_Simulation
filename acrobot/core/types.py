"""
Type definitions for the Acrobot simulation.

Lightweight types for zero-overhead state representation.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


# State type: [theta1, theta2, dtheta1, dtheta2]
State = NDArray[np.float64]

# Control input type: scalar torque
Control = float

# Time array type
TimeArray = NDArray[np.float64]

# Trajectory type: (n_steps, 4) array
Trajectory = NDArray[np.float64]


class SimulationResult(NamedTuple):
    """Immutable container for simulation output."""
    time: TimeArray           # (n_steps,) time points
    states: Trajectory        # (n_steps, 4) state trajectory
    controls: NDArray         # (n_steps,) control inputs
    energy: NDArray           # (n_steps,) total energy
    kinetic_energy: NDArray   # (n_steps,) kinetic energy
    potential_energy: NDArray  # (n_steps,) potential energy
    switch_times: NDArray     # times when controller switched
