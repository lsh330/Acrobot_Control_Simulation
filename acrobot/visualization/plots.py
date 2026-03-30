"""
Publication-quality static plots for the Acrobot simulation.

Generates:
    1. dynamics_analysis.png — State trajectories, energy, control input
    2. control_analysis.png — LQR verification, switching events
"""

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..core.types import SimulationResult
from .style import apply_publication_style


def _wrap_angle_error(states: np.ndarray, target: float = math.pi) -> np.ndarray:
    """Compute angle-wrapped error from target for theta1."""
    err = states[:, 0] - target
    return np.arctan2(np.sin(err), np.cos(err))


def plot_dynamics_analysis(
    result: SimulationResult,
    output_dir: str = "output/plots",
    dpi: int = 150,
) -> str:
    """Generate dynamics analysis figure.

    4-panel plot:
        Top-left: Joint angles (theta1, theta2)
        Top-right: Joint angular velocities
        Bottom-left: Total, kinetic, potential energy
        Bottom-right: Control torque with switching markers
    """
    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t = result.time
    theta1_err = np.degrees(_wrap_angle_error(result.states))

    # Panel 1: Angles
    ax = axes[0, 0]
    ax.plot(t, theta1_err, label=r"$\theta_1 - \pi$ (error)", color="C0")
    ax.plot(t, np.degrees(result.states[:, 1]), label=r"$\theta_2$", color="C1", alpha=0.7)
    ax.set_ylabel("Angle [deg]")
    ax.set_title("Joint Angles")
    ax.legend(loc="upper right")
    ax.axhline(0, color="k", linestyle="--", linewidth=0.5)

    # Panel 2: Velocities
    ax = axes[0, 1]
    ax.plot(t, result.states[:, 2], label=r"$\dot{\theta}_1$", color="C0")
    ax.plot(t, result.states[:, 3], label=r"$\dot{\theta}_2$", color="C1", alpha=0.7)
    ax.set_ylabel("Angular velocity [rad/s]")
    ax.set_title("Joint Velocities")
    ax.legend(loc="upper right")
    ax.axhline(0, color="k", linestyle="--", linewidth=0.5)

    # Panel 3: Energy
    ax = axes[1, 0]
    ax.plot(t, result.energy, label="Total $E$", color="C2", linewidth=2)
    ax.plot(t, result.kinetic_energy, label="Kinetic $T$", color="C0", alpha=0.6)
    ax.plot(t, result.potential_energy, label="Potential $V$", color="C1", alpha=0.6)
    ax.axhline(result.energy[0], color="gray", linestyle=":", label="$E_{down}$")
    ax.axhline(-result.energy[0], color="red", linestyle="--", linewidth=1,
               label="$E_{upright}$")
    ax.set_ylabel("Energy [J]")
    ax.set_xlabel("Time [s]")
    ax.set_title("Mechanical Energy")
    ax.legend(loc="right", fontsize=8)

    # Panel 4: Control torque
    ax = axes[1, 1]
    ax.plot(t, result.controls, color="C3", linewidth=0.8)
    for st in result.switch_times:
        ax.axvline(st, color="green", alpha=0.3, linewidth=0.5)
    ax.set_ylabel("Torque [N·m]")
    ax.set_xlabel("Time [s]")
    ax.set_title("Control Input")
    if len(result.switch_times) > 0:
        ax.axvline(result.switch_times[0], color="green", alpha=0.5,
                   label=f"LQR switch (×{len(result.switch_times)})")
        ax.legend()

    fig.suptitle("Acrobot Swing-Up + LQR Balancing: Dynamics Analysis",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = f"{output_dir}/dynamics_analysis.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_phase_portrait(
    result: SimulationResult,
    output_dir: str = "output/plots",
    dpi: int = 150,
) -> str:
    """Generate phase portrait figure.

    2-panel plot:
        Left: theta1 vs dtheta1 phase plane
        Right: theta2 vs dtheta2 phase plane
    """
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    theta1_err = _wrap_angle_error(result.states)

    # Phase plane: theta1
    ax = axes[0]
    sc = ax.scatter(np.degrees(theta1_err), result.states[:, 2],
                    c=result.time, cmap="viridis", s=0.5, alpha=0.6)
    ax.plot(0, 0, "r*", markersize=15, label="Target")
    ax.set_xlabel(r"$\theta_1 - \pi$ [deg]")
    ax.set_ylabel(r"$\dot{\theta}_1$ [rad/s]")
    ax.set_title("Phase Portrait: Link 1")
    ax.legend()
    plt.colorbar(sc, ax=ax, label="Time [s]")

    # Phase plane: theta2
    ax = axes[1]
    sc = ax.scatter(np.degrees(result.states[:, 1]), result.states[:, 3],
                    c=result.time, cmap="viridis", s=0.5, alpha=0.6)
    ax.plot(0, 0, "r*", markersize=15, label="Target")
    ax.set_xlabel(r"$\theta_2$ [deg]")
    ax.set_ylabel(r"$\dot{\theta}_2$ [rad/s]")
    ax.set_title("Phase Portrait: Link 2")
    ax.legend()
    plt.colorbar(sc, ax=ax, label="Time [s]")

    fig.suptitle("Acrobot Phase Portraits", fontsize=14, fontweight="bold")
    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = f"{output_dir}/phase_portrait.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
