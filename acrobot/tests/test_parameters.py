"""
Tests for parameter validation and derived parameter computation.
"""

import pytest

from acrobot.core.config import PhysicalParams
from acrobot.parameters.physical import get_default_params, unpack_scalars
from acrobot.parameters.derived import compute_derived, unpack_derived_scalars
from acrobot.parameters.validation import validate_params, ParameterValidationError


class TestValidation:
    def test_default_params_valid(self):
        """Default Spong/Drake parameters must pass validation."""
        validate_params(get_default_params())

    def test_negative_mass_rejected(self):
        with pytest.raises(ParameterValidationError):
            validate_params(PhysicalParams(m1=-1.0))

    def test_zero_length_rejected(self):
        with pytest.raises(ParameterValidationError):
            validate_params(PhysicalParams(l1=0.0))

    def test_com_outside_link_rejected(self):
        with pytest.raises(ParameterValidationError):
            validate_params(PhysicalParams(lc1=1.5, l1=1.0))

    def test_negative_damping_rejected(self):
        with pytest.raises(ParameterValidationError):
            validate_params(PhysicalParams(b1=-0.1))


class TestDerived:
    def test_derived_values(self):
        """Check derived parameters against manual calculation."""
        p = get_default_params()
        d = compute_derived(p)

        # alpha = Ic1 + Ic2 + m1*lc1^2 + m2*(l1^2 + lc2^2)
        alpha_expected = 0.083 + 0.083 + 1.0*0.25 + 1.0*(1.0 + 0.25)
        assert abs(d.alpha - alpha_expected) < 1e-10

        # beta = m2*l1*lc2
        assert abs(d.beta - 0.5) < 1e-10

        # delta = Ic2 + m2*lc2^2
        assert abs(d.delta - 0.333) < 1e-10

    def test_unpack_roundtrip(self):
        """Scalar unpacking preserves values."""
        p = get_default_params()
        scalars = unpack_scalars(p)
        assert len(scalars) == 11
        assert scalars[0] == p.m1

        d = compute_derived(p)
        d_scalars = unpack_derived_scalars(d)
        assert len(d_scalars) == 5
        assert d_scalars[0] == d.alpha
