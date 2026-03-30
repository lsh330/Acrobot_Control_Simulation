# Acrobot Control Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A high-performance, publication-quality simulation of the **Acrobot** (underactuated two-link planar robot) with **energy-based swing-up** and **LQR optimal balancing** control.

> **Benchmark Reference:** Spong, M.W. (1995). "The Swing Up Control Problem for the Acrobot." *IEEE Control Systems Magazine*, 15(1), 49-55.

## System Description

The Acrobot is a canonical underactuated mechanical system consisting of two serial links in a vertical plane. Only the second joint (elbow) is actuated, while the first joint (shoulder) swings freely — making it a challenging benchmark for nonlinear control.

**State vector:** `x = [theta1, theta2, dtheta1, dtheta2]`
- `theta1`: Angle of link 1 from vertical downward (counterclockwise positive)
- `theta2`: Relative angle of link 2 from link 1

**Control input:** Torque `u` applied at the elbow joint only.

## Quick Start

```bash
# Clone
git clone https://github.com/lsh330/Acrobot_Control_Simulation.git
cd Acrobot_Control_Simulation

# Install dependencies
pip install -r requirements.txt

# Run simulation with default parameters
python main.py

# Run with custom config
python main.py --config config.yaml

# Run with CLI overrides
python main.py --method hybrid --t-final 15 --save-gif
```

## Project Structure

```
acrobot/
├── core/               # Configuration, constants, type definitions
├── parameters/         # Physical and derived system parameters
├── dynamics/           # Lagrangian mechanics, EOM, Jacobians
├── linearization/      # Equilibrium analysis, state-space models
├── control/            # LQR, energy swing-up, hybrid controller
├── simulation/         # Numerical integrators, simulation engine
├── analysis/           # Stability, frequency response, ROA, robustness
├── visualization/      # Publication-quality plots and animations
├── tests/              # Comprehensive test suite
├── pipeline/           # Workflow orchestration
└── utils/              # Performance timing, logging
```

## Theoretical Background

*Section will be expanded with simulation results and analysis plots.*

## References

1. Spong, M.W. (1995). "The Swing Up Control Problem for the Acrobot." *IEEE Control Systems Magazine*, 15(1), 49-55.
2. Xin, X. & Kaneda, M. (2007). "Analysis of the Energy-Based Swing-Up Control of the Acrobot." *Int. J. Robust Nonlinear Control*, 17(16), 1503-1524.
3. Tedrake, R. *Underactuated Robotics.* MIT OpenCourseWare.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
