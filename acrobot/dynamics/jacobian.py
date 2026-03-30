"""
Analytical Jacobian (linearization matrices A, B) of the Acrobot.

Linearization about an arbitrary state x0 = [q1, q2, dq1, dq2]:
    dx/dt ≈ A*(x - x0) + B*(u - u0)

For the upright equilibrium x_eq = [pi, 0, 0, 0], u_eq = 0:
    Small-angle approximation: sin(pi+d1) = -sin(d1) ≈ -d1
                                cos(pi+d1) = -cos(d1) ≈ -1
                                sin(d2) ≈ d2
                                cos(d2) ≈ 1

The analytical Jacobian avoids numerical differentiation entirely,
yielding ~100x speedup and eliminating finite-difference truncation error.

State-space form:
    x_dot = [dq1, dq2, f3(x,u), f4(x,u)]^T

    A = [[0, 0, 1, 0],
         [0, 0, 0, 1],
         [df3/dq1, df3/dq2, df3/ddq1, df3/ddq2],
         [df4/dq1, df4/dq2, df4/ddq1, df4/ddq2]]

    B = [[0], [0], [df3/du], [df4/du]]

Reference:
    Tedrake, R. Underactuated Robotics, Ch. 3 (Acrobot).
"""

import math

import numpy as np
import numba as nb


@nb.njit(cache=True, fastmath=True)
def linearize_at_upright(
    alpha: float,
    beta: float,
    delta: float,
    phi1: float,
    phi2: float,
    b1: float,
    b2: float,
) -> tuple:
    """Compute A, B matrices at the upright equilibrium [pi, 0, 0, 0].

    At the upright equilibrium with cos(theta2)=1:
        M_eq = [[alpha + 2*beta, delta + beta],
                [delta + beta,   delta       ]]

    Gravity Jacobian at upright (sin(pi)=0, cos(pi)=-1):
        dG1/dq1 = -phi1 - phi2  (destabilizing)
        dG1/dq2 = -phi2
        dG2/dq1 = -phi2
        dG2/dq2 = -phi2

    Returns:
        (A_flat, B_flat) as 1D arrays of length 16 and 4 respectively.
        Reshape A_flat to (4,4), B_flat to (4,1).
    """
    # Mass matrix at equilibrium (theta2=0 → cos(theta2)=1)
    d11 = alpha + 2.0 * beta
    d12 = delta + beta
    d22 = delta

    # M^{-1}
    det = d11 * d22 - d12 * d12
    inv_det = 1.0 / det
    mi11 = d22 * inv_det
    mi12 = -d12 * inv_det
    mi22 = d11 * inv_det

    # Gravity Jacobian at upright: dG/dq
    # At theta1=pi, theta2=0:
    # dG1/dtheta1 = phi1*cos(pi) + phi2*cos(pi) = -(phi1+phi2)
    # dG1/dtheta2 = phi2*cos(pi+0) = -phi2
    # dG2/dtheta1 = phi2*cos(pi+0) = -phi2
    # dG2/dtheta2 = phi2*cos(pi+0) = -phi2
    dG1_dq1 = -(phi1 + phi2)
    dG1_dq2 = -phi2
    dG2_dq1 = -phi2
    dG2_dq2 = -phi2

    # A matrix (4x4)
    # Row 0,1: dx1/dt = dq1, dx2/dt = dq2 (kinematic)
    # Row 2,3: ddq = M^{-1}*(-dG/dq * delta_q - D*dq)
    # df3/dq1 = -(mi11*dG1_dq1 + mi12*dG2_dq1)
    # df3/dq2 = -(mi11*dG1_dq2 + mi12*dG2_dq2)
    # df3/ddq1 = -(mi11*b1)
    # df3/ddq2 = -(mi12*b2)
    # Similarly for f4

    A = np.zeros(16, dtype=np.float64)
    # Row 0: [0, 0, 1, 0]
    A[2] = 1.0
    # Row 1: [0, 0, 0, 1]
    A[7] = 1.0
    # Row 2: df3/d(state)
    A[8] = -(mi11 * dG1_dq1 + mi12 * dG2_dq1)
    A[9] = -(mi11 * dG1_dq2 + mi12 * dG2_dq2)
    A[10] = -(mi11 * b1)
    A[11] = -(mi12 * b2)
    # Row 3: df4/d(state)
    A[12] = -(mi12 * dG1_dq1 + mi22 * dG2_dq1)
    A[13] = -(mi12 * dG1_dq2 + mi22 * dG2_dq2)
    A[14] = -(mi12 * b1)
    A[15] = -(mi22 * b2)

    # B matrix (4x1): df/du
    # B_acrobot = [0, 1]^T, so effect of u:
    # ddq = M^{-1}*[0, u]^T → df3/du = mi12, df4/du = mi22
    B = np.zeros(4, dtype=np.float64)
    B[2] = mi12
    B[3] = mi22

    return A, B


def linearize_at_upright_matrices(
    alpha: float,
    beta: float,
    delta: float,
    phi1: float,
    phi2: float,
    b1: float,
    b2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning properly shaped numpy arrays.

    Returns:
        (A, B) where A is (4,4) and B is (4,1).
    """
    A_flat, B_flat = linearize_at_upright(
        alpha, beta, delta, phi1, phi2, b1, b2)
    A = A_flat.reshape(4, 4)
    B = B_flat.reshape(4, 1)
    return A, B
