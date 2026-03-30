"""
System configuration dataclass.

Immutable (frozen) configuration that holds all simulation parameters.
Supports construction from YAML files and CLI argument overrides.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .constants import GRAVITY


@dataclass(frozen=True, slots=True)
class PhysicalParams:
    """Immutable physical parameters of the Acrobot system.

    Default values follow the Spong/Drake canonical benchmark:
    Spong, M.W. (1995). IEEE Control Systems Magazine, 15(1), 49-55.
    """
    m1: float = 1.0       # Link 1 mass [kg]
    m2: float = 1.0       # Link 2 mass [kg]
    l1: float = 1.0       # Link 1 length [m]
    l2: float = 1.0       # Link 2 length [m]
    lc1: float = 0.5      # Link 1 center of mass distance [m]
    lc2: float = 0.5      # Link 2 center of mass distance [m]
    Ic1: float = 0.083    # Link 1 moment of inertia about CoM [kg*m^2]
    Ic2: float = 0.083    # Link 2 moment of inertia about CoM [kg*m^2]
    b1: float = 0.1       # Joint 1 viscous damping [N*m*s/rad]
    b2: float = 0.1       # Joint 2 viscous damping [N*m*s/rad]
    g: float = GRAVITY    # Gravitational acceleration [m/s^2]


@dataclass(frozen=True, slots=True)
class SimulationParams:
    """Immutable simulation parameters."""
    dt: float = 0.001             # Time step [s]
    t_final: float = 60.0        # Total duration [s]
    integrator: str = "rk4"      # Integration method
    initial_state: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class ControlParams:
    """Immutable control parameters."""
    method: str = "hybrid"
    Q_diag: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    R: float = 1.0
    k_energy: float = 5.0
    switching_radius: float = 10.0    # Lyapunov distance threshold
    hysteresis: float = 5.0           # Multiplier for deactivation band
    max_torque: float = 20.0


@dataclass(frozen=True, slots=True)
class AnalysisParams:
    """Immutable analysis parameters."""
    roa_samples: int = 1000
    robustness_variation: float = 0.2
    robustness_trials: int = 200


@dataclass(frozen=True, slots=True)
class VisualizationParams:
    """Immutable visualization parameters."""
    save_plots: bool = True
    save_gif: bool = False
    fps: int = 30
    dpi: int = 150
    output_dir: str = "output"


@dataclass(frozen=True, slots=True)
class SystemConfig:
    """Top-level immutable system configuration."""
    physical: PhysicalParams = field(default_factory=PhysicalParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)
    control: ControlParams = field(default_factory=ControlParams)
    analysis: AnalysisParams = field(default_factory=AnalysisParams)
    visualization: VisualizationParams = field(default_factory=VisualizationParams)

    @staticmethod
    def from_yaml(path: str | Path) -> "SystemConfig":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        physical = PhysicalParams(**raw.get("physical", {}))
        sim_raw = raw.get("simulation", {})
        if "initial_state" in sim_raw:
            sim_raw["initial_state"] = tuple(sim_raw["initial_state"])
        simulation = SimulationParams(**sim_raw)
        ctrl_raw = raw.get("control", {})
        if "Q_diag" in ctrl_raw:
            ctrl_raw["Q_diag"] = tuple(ctrl_raw["Q_diag"])
        control = ControlParams(**ctrl_raw)
        analysis = AnalysisParams(**raw.get("analysis", {}))
        visualization = VisualizationParams(**raw.get("visualization", {}))

        return SystemConfig(
            physical=physical,
            simulation=simulation,
            control=control,
            analysis=analysis,
            visualization=visualization,
        )

    @staticmethod
    def from_args(args, base: Optional["SystemConfig"] = None) -> "SystemConfig":
        """Override config with CLI arguments."""
        cfg = base or SystemConfig()
        overrides = {}
        if hasattr(args, "method") and args.method:
            overrides["control"] = ControlParams(
                method=args.method,
                Q_diag=cfg.control.Q_diag,
                R=cfg.control.R,
                k_energy=cfg.control.k_energy,
                switching_radius=cfg.control.switching_radius,
                hysteresis=cfg.control.hysteresis,
                max_torque=cfg.control.max_torque,
            )
        if hasattr(args, "t_final") and args.t_final:
            overrides["simulation"] = SimulationParams(
                dt=args.dt if hasattr(args, "dt") and args.dt else cfg.simulation.dt,
                t_final=args.t_final,
                integrator=cfg.simulation.integrator,
                initial_state=cfg.simulation.initial_state,
            )
        if hasattr(args, "output_dir") and args.output_dir:
            overrides["visualization"] = VisualizationParams(
                save_plots=cfg.visualization.save_plots,
                save_gif=getattr(args, "save_gif", cfg.visualization.save_gif),
                fps=cfg.visualization.fps,
                dpi=cfg.visualization.dpi,
                output_dir=args.output_dir,
            )
        return SystemConfig(
            physical=cfg.physical,
            simulation=overrides.get("simulation", cfg.simulation),
            control=overrides.get("control", cfg.control),
            analysis=cfg.analysis,
            visualization=overrides.get("visualization", cfg.visualization),
        )
