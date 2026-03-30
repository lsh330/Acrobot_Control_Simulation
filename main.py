"""
Acrobot Control Simulation - Entry Point

High-performance simulation of the Acrobot with energy-based swing-up
and LQR optimal balancing control.

Usage:
    python main.py                          # Default parameters
    python main.py --config config.yaml     # Custom config
    python main.py --method hybrid --t-final 15
"""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Acrobot Control Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file")
    parser.add_argument("--method", type=str, default="hybrid",
                        choices=["lqr", "energy_swing_up", "hybrid"],
                        help="Control method")
    parser.add_argument("--t-final", type=float, default=10.0,
                        help="Simulation duration [s]")
    parser.add_argument("--dt", type=float, default=0.001,
                        help="Integration time step [s]")
    parser.add_argument("--save-plots", action="store_true", default=True,
                        help="Save analysis plots")
    parser.add_argument("--save-gif", action="store_true", default=False,
                        help="Save animation GIF")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory")
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    # Pipeline orchestration will be implemented in pipeline/orchestrator.py
    print("Acrobot Control Simulation")
    print(f"  Method: {args.method}")
    print(f"  Duration: {args.t_final}s, dt={args.dt}s")
    print("  (Implementation in progress)")


if __name__ == "__main__":
    main()
