"""
Energy-based swing-up controller for the Acrobot.

Combines Drake/Spong energy shaping with collocated Partial Feedback
Linearization (PFL) for theta2 regulation.

Control law: u = u_e + u_p

    u_e = -k_e * (E - E_d) * dtheta2        (energy shaping)
    u_p = a3*y - a2*f1 + f2                  (collocated PFL)

The PFL term cancels the nonlinear dynamics of the actuated joint
and replaces them with a linear PD controller:
    ddtheta2 = y = -k_p * theta2 - k_d * dtheta2

Derivation of PFL:
    EOM row 1: d11*ddq1 + d12*ddq2 = -f1
    EOM row 2: d12*ddq1 + d22*ddq2 = u - f2
    where f1 = coriolis_1 + gravity_1, f2 = coriolis_2 + gravity_2

    Eliminating ddq1:
        a3*ddq2 = u - f2 + a2*f1
    where a2 = d12/d11, a3 = d22 - d12^2/d11

    Setting ddq2 = y:
        u_p = a3*y + f2 - a2*f1

Dead-start handler: applies constant torque when the system is at rest.

Reference:
    Spong, M.W. (1995). IEEE CSM 15(1), 49-55.
    Drake: AcrobotSpongController.
"""

import math

import numba as nb

from ..dynamics.energy import total_energy, upright_energy


@nb.njit(cache=True, fastmath=True)
def energy_swing_up_control(
    theta1: float,
    theta2: float,
    dtheta1: float,
    dtheta2: float,
    k_energy: float,
    max_torque: float,
    alpha: float,
    beta: float,
    delta: float,
    phi1: float,
    phi2: float,
) -> float:
    """Compute swing-up torque: energy shaping + collocated PFL."""
    E = total_energy(theta1, theta2, dtheta1, dtheta2,
                     alpha, beta, delta, phi1, phi2)
    E_d = upright_energy(phi1, phi2)
    E_tilde = E - E_d

    # --- Dead-start handler ---
    speed = abs(dtheta1) + abs(dtheta2)
    if speed < 0.01 and abs(E_tilde) > 0.1 * E_d:
        u = max_torque
        return u

    # --- Mass matrix elements ---
    c2 = math.cos(theta2)
    s2 = math.sin(theta2)
    d11 = alpha + 2.0 * beta * c2
    d12 = delta + beta * c2
    d22 = delta

    # --- Bias terms (Coriolis + gravity) ---
    s1 = math.sin(theta1)
    s12 = math.sin(theta1 + theta2)
    h = -beta * s2

    # f1: row-1 bias (unactuated joint)
    # tau_c1 + g1 = h*dtheta2*(2*dtheta1+dtheta2) + phi1*sin(theta1) + phi2*sin(theta1+theta2)
    f1 = h * dtheta2 * (2.0 * dtheta1 + dtheta2) + phi1 * s1 + phi2 * s12

    # f2: row-2 bias (actuated joint, without control input)
    # tau_c2 + g2 = -h*dtheta1^2 + phi2*sin(theta1+theta2)
    f2 = -h * dtheta1 * dtheta1 + phi2 * s12

    # --- PFL coefficients ---
    a2 = d12 / d11
    a3 = d22 - d12 * d12 / d11   # Schur complement, always > 0

    # --- Energy shaping: u_e ---
    u_e = -k_energy * E_tilde * dtheta2

    # --- Collocated PFL: u_p ---
    # Desired theta2 acceleration: y = -kp*theta2 - kd*dtheta2
    k_p = 10.0 * k_energy   # 50.0 when k_energy=5.0
    k_d = 1.0 * k_energy    # 5.0 when k_energy=5.0

    y = -k_p * theta2 - k_d * dtheta2

    # PFL torque: u_p = a3*y + f2 - a2*f1
    u_p = a3 * y + f2 - a2 * f1

    # --- Combined ---
    u = u_e + u_p

    # --- Saturation ---
    if u > max_torque:
        return max_torque
    if u < -max_torque:
        return -max_torque
    return u
