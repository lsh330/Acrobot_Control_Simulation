"""
Performance metrics for the Acrobot simulation.

Computes swing-up time, settling time, overshoot, and other metrics.
"""

import math

import numpy as np

from ..core.types import SimulationResult


def compute_metrics(result: SimulationResult) -> dict:
    """Compute performance metrics from simulation result.

    Returns dict with:
        swing_up_time: Time to first reach near upright [s]
        settling_time: Time after which error stays below threshold [s]
        max_overshoot: Maximum energy overshoot above E_d [J]
        mean_control_effort: RMS control torque [N*m]
        total_switches: Number of controller switches
    """
    threshold_rad = 0.1  # 5.7 degrees

    # Angle error from upright
    errors = np.array([
        abs(math.atan2(math.sin(s[0] - math.pi), math.cos(s[0] - math.pi)))
        for s in result.states
    ])

    # Swing-up time: first time error < threshold
    swing_up_time = None
    for i, e in enumerate(errors):
        if e < threshold_rad:
            swing_up_time = result.time[i]
            break

    # Settling time: last time error exceeds threshold
    settling_time = None
    for i in range(len(errors) - 1, -1, -1):
        if errors[i] > threshold_rad:
            if i + 1 < len(errors):
                settling_time = result.time[i + 1]
            break

    # Energy overshoot
    E_d = -result.energy[0]  # upright energy ≈ -E_down
    max_overshoot = float(np.max(result.energy) - E_d) if E_d > 0 else 0.0

    # Control effort
    mean_effort = float(np.sqrt(np.mean(result.controls**2)))
    max_torque = float(np.max(np.abs(result.controls)))

    return {
        "swing_up_time": swing_up_time,
        "settling_time": settling_time,
        "max_angle_error_deg": float(np.max(errors) * 180 / math.pi),
        "final_angle_error_deg": float(errors[-1] * 180 / math.pi),
        "max_energy_overshoot": max_overshoot,
        "rms_control_effort": mean_effort,
        "max_control_torque": max_torque,
        "total_switches": len(result.switch_times),
    }
