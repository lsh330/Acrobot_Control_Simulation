"""
Acrobot Control Simulation - Entry Point

High-performance simulation of the Acrobot with energy-based swing-up
and LQR optimal balancing control.

Usage:
    python main.py                          # Default parameters
    python main.py --config config.yaml     # Custom config
    python main.py --method hybrid --t-final 60 --save-gif
"""

import argparse
import sys

from acrobot.core.config import SystemConfig
from acrobot.pipeline.orchestrator import run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Acrobot Control Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file")
    parser.add_argument("--method", type=str, default=None,
                        choices=["lqr", "energy_swing_up", "hybrid"],
                        help="Control method")
    parser.add_argument("--t-final", type=float, default=None,
                        help="Simulation duration [s]")
    parser.add_argument("--dt", type=float, default=None,
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

    # Build configuration
    if args.config:
        config = SystemConfig.from_yaml(args.config)
    else:
        config = SystemConfig()

    config = SystemConfig.from_args(args, config)

    # Run pipeline
    results = run_pipeline(config)

    # Summary
    m = results["metrics"]
    print("\n" + "=" * 50)
    print("  SIMULATION RESULTS")
    print("=" * 50)
    print(f"  Swing-up time:    {m['swing_up_time']}s")
    print(f"  Settling time:    {m['settling_time']}s")
    print(f"  Final error:      {m['final_angle_error_deg']:.4f} deg")
    print(f"  RMS torque:       {m['rms_control_effort']:.2f} N*m")
    print(f"  Controller switches: {m['total_switches']}")
    print(f"  Simulation time:  {results['sim_time']:.3f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
