"""
Tests for the Acrobot control module.

Validates:
    - LQR design produces stable closed-loop
    - Controllability at upright equilibrium
    - CARE solution properties
    - Hybrid controller swing-up success
"""

import math

import numpy as np
import pytest

from acrobot.core.config import PhysicalParams, ControlParams, SystemConfig, SimulationParams
from acrobot.control.lqr import design_lqr
from acrobot.control.hybrid import HybridController
from acrobot.linearization.controllability import is_controllable
from acrobot.linearization.state_space import get_state_space
from acrobot.simulation.runner import run_simulation


@pytest.fixture
def params():
    return PhysicalParams()


@pytest.fixture
def ctrl():
    return ControlParams()


class TestLQR:
    def test_controllable(self, params):
        """System must be controllable at upright."""
        A, B = get_state_space(params)
        assert is_controllable(A, B)

    def test_stable_eigenvalues(self, params, ctrl):
        """All closed-loop eigenvalues must have negative real parts."""
        K, P, eigvals = design_lqr(params, ctrl)
        assert all(e.real < 0 for e in eigvals)

    def test_care_solution_symmetric(self, params, ctrl):
        """P matrix from CARE must be symmetric."""
        K, P, _ = design_lqr(params, ctrl)
        np.testing.assert_allclose(P, P.T, atol=1e-10)

    def test_care_solution_positive_definite(self, params, ctrl):
        """P matrix must be positive definite."""
        K, P, _ = design_lqr(params, ctrl)
        eigvals_P = np.linalg.eigvals(P)
        assert all(e > 0 for e in eigvals_P)

    def test_care_residual(self, params, ctrl):
        """CARE residual should be near machine epsilon."""
        A, B = get_state_space(params)
        K, P, _ = design_lqr(params, ctrl)
        Q = np.diag(ctrl.Q_diag)
        R = np.array([[ctrl.R]])

        residual = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q
        assert np.linalg.norm(residual) < 1e-6


class TestHybrid:
    def test_swing_up_success(self):
        """Full swing-up + balancing should converge within 60s."""
        config = SystemConfig(
            simulation=SimulationParams(dt=0.001, t_final=60.0),
            control=ControlParams(
                k_energy=5.0, switching_radius=10.0,
                hysteresis=5.0, max_torque=20.0),
        )
        result = run_simulation(config)

        # Check final state is near upright
        final_err = abs(math.atan2(
            math.sin(result.states[-1, 0] - math.pi),
            math.cos(result.states[-1, 0] - math.pi)))
        assert final_err < 0.01, f"Final error {final_err} rad too large"

        # Check energy converged
        E_target = result.energy[0] * -1  # upright energy
        assert abs(result.energy[-1] - E_target) < 0.1
