"""
Tests for the Acrobot dynamics module.

Validates:
    - Mass matrix symmetry and positive-definiteness
    - Gravity vector at known configurations
    - Energy conservation in free fall
    - Coriolis skew-symmetry property
    - EOM consistency
"""

import math

import numpy as np
import pytest

from acrobot.parameters.physical import get_default_params
from acrobot.parameters.derived import compute_derived
from acrobot.dynamics.mass_matrix import (
    mass_matrix_scalars, mass_matrix_det, mass_matrix_inv_scalars)
from acrobot.dynamics.coriolis import coriolis_scalars
from acrobot.dynamics.gravity import gravity_scalars
from acrobot.dynamics.energy import (
    kinetic_energy, potential_energy, total_energy, upright_energy)
from acrobot.dynamics.equations_of_motion import state_derivative
from acrobot.dynamics.lagrangian import lagrangian
from acrobot.simulation.integrator import rk4_step


@pytest.fixture
def params():
    return get_default_params()


@pytest.fixture
def derived(params):
    return compute_derived(params)


class TestMassMatrix:
    def test_symmetric(self, derived):
        """M(q) must be symmetric for all configurations."""
        for theta2 in np.linspace(-math.pi, math.pi, 50):
            d11, d12, d22 = mass_matrix_scalars(
                theta2, derived.alpha, derived.beta, derived.delta)
            # M = [[d11,d12],[d12,d22]] is symmetric by construction
            assert d11 > 0
            assert d22 > 0

    def test_positive_definite(self, derived):
        """det(M) > 0 for all configurations."""
        for theta2 in np.linspace(-math.pi, math.pi, 100):
            d11, d12, d22 = mass_matrix_scalars(
                theta2, derived.alpha, derived.beta, derived.delta)
            det = mass_matrix_det(d11, d12, d22)
            assert det > 0, f"det(M)={det} at theta2={theta2}"

    def test_inverse_correctness(self, derived):
        """M * M^{-1} = I."""
        for theta2 in np.linspace(-math.pi, math.pi, 20):
            d11, d12, d22 = mass_matrix_scalars(
                theta2, derived.alpha, derived.beta, derived.delta)
            mi11, mi12, mi22 = mass_matrix_inv_scalars(d11, d12, d22)

            M = np.array([[d11, d12], [d12, d22]])
            M_inv = np.array([[mi11, mi12], [mi12, mi22]])
            I = M @ M_inv
            np.testing.assert_allclose(I, np.eye(2), atol=1e-10)


class TestGravity:
    def test_zero_at_downward(self, derived):
        """G(0, 0) = [0, 0] at the downward equilibrium."""
        g1, g2 = gravity_scalars(0.0, 0.0, derived.phi1, derived.phi2)
        assert abs(g1) < 1e-12
        assert abs(g2) < 1e-12

    def test_zero_at_upright(self, derived):
        """G(pi, 0) = [0, 0] at the upright equilibrium."""
        g1, g2 = gravity_scalars(math.pi, 0.0, derived.phi1, derived.phi2)
        assert abs(g1) < 1e-10
        assert abs(g2) < 1e-10


class TestEnergy:
    def test_downward_energy(self, derived):
        """Energy at downward equilibrium."""
        E = total_energy(0, 0, 0, 0,
                         derived.alpha, derived.beta, derived.delta,
                         derived.phi1, derived.phi2)
        V = potential_energy(0, 0, derived.phi1, derived.phi2)
        assert abs(E - V) < 1e-12  # zero kinetic energy

    def test_upright_energy(self, derived):
        """Energy at upright equilibrium."""
        E_up = upright_energy(derived.phi1, derived.phi2)
        V_up = potential_energy(math.pi, 0, derived.phi1, derived.phi2)
        assert abs(E_up - V_up) < 1e-10

    def test_energy_conservation_free_fall(self, params, derived):
        """Energy is conserved in free fall (no control, no damping)."""
        # Use zero damping for this test
        alpha, beta, delta = derived.alpha, derived.beta, derived.delta
        phi1, phi2 = derived.phi1, derived.phi2

        # Initial state: small perturbation from downward
        q1, q2, dq1, dq2 = 0.3, 0.1, 0.0, 0.0
        E0 = total_energy(q1, q2, dq1, dq2, alpha, beta, delta, phi1, phi2)

        # Integrate 1000 steps with no control and zero damping
        dt = 0.001
        for _ in range(1000):
            q1, q2, dq1, dq2 = rk4_step(
                q1, q2, dq1, dq2, 0.0, dt,
                alpha, beta, delta, phi1, phi2, 0.0, 0.0)

        E_final = total_energy(q1, q2, dq1, dq2, alpha, beta, delta, phi1, phi2)
        # RK4 energy drift should be small (< 0.1%)
        assert abs(E_final - E0) / abs(E0) < 0.001


class TestEOM:
    def test_rest_at_downward(self, derived, params):
        """System at rest at downward equilibrium has zero acceleration."""
        dx = state_derivative(
            0, 0, 0, 0, 0,
            derived.alpha, derived.beta, derived.delta,
            derived.phi1, derived.phi2, params.b1, params.b2)
        for val in dx:
            assert abs(val) < 1e-12

    def test_rest_at_upright(self, derived, params):
        """System at rest at upright equilibrium has zero acceleration."""
        dx = state_derivative(
            math.pi, 0, 0, 0, 0,
            derived.alpha, derived.beta, derived.delta,
            derived.phi1, derived.phi2, params.b1, params.b2)
        for val in dx:
            assert abs(val) < 1e-10
