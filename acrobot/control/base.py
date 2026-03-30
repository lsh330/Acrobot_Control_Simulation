"""
Abstract base class for Acrobot controllers.

Defines the interface that all controllers must implement,
enabling clean dependency injection and strategy pattern.
"""

from abc import ABC, abstractmethod

import numpy as np


class ControllerBase(ABC):
    """Abstract controller interface.

    All Acrobot controllers (LQR, energy swing-up, hybrid)
    implement this interface for interchangeability.
    """

    __slots__ = ()

    @abstractmethod
    def compute_control(self, state: np.ndarray) -> float:
        """Compute control torque for the current state.

        Args:
            state: Current state [theta1, theta2, dtheta1, dtheta2]

        Returns:
            Control torque u [N*m]
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset controller internal state."""
        ...

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Current controller mode name (for logging/display)."""
        ...
