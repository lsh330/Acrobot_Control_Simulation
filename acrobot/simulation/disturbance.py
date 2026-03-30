"""
Disturbance generation for robustness testing.

Provides band-limited white noise and impulse disturbances
for evaluating controller performance under perturbations.
"""

import numpy as np
from scipy.signal import butter, lfilter


def band_limited_noise(
    n_steps: int,
    dt: float,
    amplitude: float = 0.5,
    cutoff_hz: float = 3.0,
    order: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Generate band-limited white noise disturbance.

    Uses a Butterworth low-pass filter to limit the bandwidth.

    Args:
        n_steps: Number of time steps
        dt: Time step [s]
        amplitude: RMS amplitude of the noise
        cutoff_hz: Cutoff frequency [Hz]
        order: Butterworth filter order
        seed: Random seed for reproducibility

    Returns:
        (n_steps,) disturbance array
    """
    rng = np.random.default_rng(seed)
    white = rng.normal(0, amplitude, n_steps)

    nyquist = 0.5 / dt
    if cutoff_hz >= nyquist:
        return white

    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    return lfilter(b, a, white).astype(np.float64)


def impulse_disturbance(
    n_steps: int,
    impulse_step: int,
    magnitude: float = 5.0,
) -> np.ndarray:
    """Generate an impulse disturbance at a specific time step.

    Args:
        n_steps: Total number of time steps
        impulse_step: Step index for the impulse
        magnitude: Impulse magnitude [N*m]

    Returns:
        (n_steps,) disturbance array
    """
    d = np.zeros(n_steps, dtype=np.float64)
    if 0 <= impulse_step < n_steps:
        d[impulse_step] = magnitude
    return d
