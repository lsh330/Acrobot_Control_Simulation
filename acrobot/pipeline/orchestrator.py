"""
Pipeline orchestrator for the Acrobot simulation.

Runs the complete workflow: validation → simulation → analysis → visualization.
"""

import time
from pathlib import Path

from ..core.config import SystemConfig
from ..parameters.validation import validate_params
from ..parameters.derived import compute_derived
from ..simulation.runner import run_simulation
from ..analysis.stability import verify_stability
from ..analysis.performance import compute_metrics
from ..utils.logger import info

# Visualization modules are lazy-imported to avoid matplotlib/PIL
# overhead when plots/GIF are disabled (~100ms startup savings)


def run_pipeline(config: SystemConfig) -> dict:
    """Execute the complete simulation pipeline.

    Steps:
        1. Validate parameters
        2. Run simulation
        3. Verify LQR stability
        4. Compute performance metrics
        5. Generate visualization outputs

    Returns:
        dict with all results and output paths.
    """
    output_dir = config.visualization.output_dir
    plots_dir = f"{output_dir}/plots"
    gifs_dir = f"{output_dir}/gifs"

    # 1. Validate
    info("Validating parameters...")
    validate_params(config.physical)
    derived = compute_derived(config.physical)
    info(f"  alpha={derived.alpha:.4f}, beta={derived.beta:.4f}, "
         f"delta={derived.delta:.4f}")

    # 2. Simulate
    info(f"Running simulation ({config.simulation.t_final}s, "
         f"dt={config.simulation.dt}s)...")
    t0 = time.perf_counter()
    result = run_simulation(config)
    sim_time = time.perf_counter() - t0
    info(f"  Completed in {sim_time:.3f}s "
         f"({len(result.time)} steps)")

    # 3. Stability verification
    info("Verifying LQR stability...")
    stability = verify_stability(config.physical, config.control)
    info(f"  All stable: {stability['all_stable']}")
    info(f"  CARE residual: {stability['care_residual_norm']:.2e}")

    # 4. Performance metrics
    info("Computing performance metrics...")
    metrics = compute_metrics(result)
    info(f"  Swing-up time: {metrics['swing_up_time']}s")
    info(f"  Settling time: {metrics['settling_time']}s")
    info(f"  Final error: {metrics['final_angle_error_deg']:.4f} deg")

    # 5. Visualization
    outputs = {}

    if config.visualization.save_plots:
        info("Generating plots...")
        from ..visualization.plots import plot_dynamics_analysis, plot_phase_portrait
        outputs["dynamics_plot"] = plot_dynamics_analysis(
            result, plots_dir, config.visualization.dpi)
        outputs["phase_plot"] = plot_phase_portrait(
            result, plots_dir, config.visualization.dpi)
        info(f"  Saved to {plots_dir}/")

    if config.visualization.save_gif:
        info("Generating animation GIF...")
        from ..visualization.animation import generate_animation
        outputs["animation"] = generate_animation(
            result, config.physical, gifs_dir,
            config.visualization.fps, dpi=80)
        info(f"  Saved to {gifs_dir}/")

    return {
        "result": result,
        "stability": stability,
        "metrics": metrics,
        "outputs": outputs,
        "sim_time": sim_time,
    }
