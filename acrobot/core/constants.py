"""
Physical constants used throughout the simulation.

All values are in SI units.
"""

# Gravitational acceleration [m/s^2]
GRAVITY: float = 9.81

# Numerical tolerances
EPS: float = 1e-12          # Machine-epsilon-scale tolerance
ENERGY_TOL: float = 1e-6    # Energy conservation tolerance
SYMM_TOL: float = 1e-10     # Symmetry check tolerance
PD_TOL: float = 1e-10       # Positive-definiteness tolerance

# Default simulation bounds
MAX_ANGULAR_VEL: float = 20.0   # Maximum angular velocity [rad/s]
MAX_TORQUE: float = 20.0        # Maximum control torque [N*m]
