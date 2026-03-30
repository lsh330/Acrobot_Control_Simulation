"""
Simulation runner for the Acrobot.

Orchestrates the time-stepping loop, applying the controller
at each step and recording the full trajectory.

Pre-allocates all output arrays for zero-allocation during the loop.
"""

import math

import numpy as np

from ..core.config import SystemConfig
from ..core.types import SimulationResult
from ..parameters.derived import compute_derived
from ..dynamics.energy import kinetic_energy, potential_energy, total_energy
from ..simulation.integrator import rk4_step
from ..control.hybrid import HybridController


def run_simulation(config: SystemConfig) -> SimulationResult:
    """Run the full Acrobot simulation.

    Args:
        config: Complete system configuration.

    Returns:
        SimulationResult with full trajectory data.
    """
    p = config.physical
    sim = config.simulation
    ctrl_cfg = config.control
    d = compute_derived(p)

    # Pre-allocate arrays
    n_steps = int(sim.t_final / sim.dt) + 1
    time = np.linspace(0, sim.t_final, n_steps)
    states = np.empty((n_steps, 4), dtype=np.float64)
    controls = np.empty(n_steps, dtype=np.float64)
    energy = np.empty(n_steps, dtype=np.float64)
    ke = np.empty(n_steps, dtype=np.float64)
    pe = np.empty(n_steps, dtype=np.float64)
    switch_list: list[float] = []

    # Initial state
    states[0] = np.array(sim.initial_state, dtype=np.float64)

    # Initialize controller
    controller = HybridController(p, ctrl_cfg)
    prev_lqr = controller.is_using_lqr

    # Record initial energy
    s = states[0]
    ke[0] = kinetic_energy(s[1], s[2], s[3], d.alpha, d.beta, d.delta)
    pe[0] = potential_energy(s[0], s[1], d.phi1, d.phi2)
    energy[0] = ke[0] + pe[0]

    # Warm up Numba JIT (first call triggers compilation)
    _ = rk4_step(s[0], s[1], s[2], s[3], 0.0, sim.dt,
                 d.alpha, d.beta, d.delta, d.phi1, d.phi2, p.b1, p.b2)

    # Main simulation loop
    for i in range(n_steps - 1):
        s = states[i]

        # Compute control
        u = controller.compute_control(s)
        controls[i] = u

        # Detect switching events
        if controller.is_using_lqr != prev_lqr:
            switch_list.append(time[i])
            prev_lqr = controller.is_using_lqr

        # Integrate one step
        q1, q2, dq1, dq2 = rk4_step(
            s[0], s[1], s[2], s[3], u, sim.dt,
            d.alpha, d.beta, d.delta, d.phi1, d.phi2, p.b1, p.b2)

        states[i + 1] = (q1, q2, dq1, dq2)

        # Record energy
        ke[i + 1] = kinetic_energy(q2, dq1, dq2, d.alpha, d.beta, d.delta)
        pe[i + 1] = potential_energy(q1, q2, d.phi1, d.phi2)
        energy[i + 1] = ke[i + 1] + pe[i + 1]

    # Final control value
    controls[-1] = controller.compute_control(states[-1])

    return SimulationResult(
        time=time,
        states=states,
        controls=controls,
        energy=energy,
        kinetic_energy=ke,
        potential_energy=pe,
        switch_times=np.array(switch_list, dtype=np.float64),
    )
