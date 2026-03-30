"""
Acrobot animation generator.

Creates a GIF animation showing the physical motion of the Acrobot
pendulum during the swing-up and balancing simulation.
"""

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from ..core.config import PhysicalParams
from ..core.types import SimulationResult


def generate_animation(
    result: SimulationResult,
    physical: PhysicalParams,
    output_dir: str = "output/gifs",
    fps: int = 30,
    dpi: int = 100,
) -> str:
    """Generate Acrobot animation GIF.

    Renders the two-link pendulum at each frame, showing:
    - Link positions with pivot/joint/tip markers
    - Time, energy, and control mode overlay
    - Trail of the tip position

    Args:
        result: Simulation result data
        physical: Physical parameters for link lengths
        output_dir: Output directory
        fps: Frames per second
        dpi: Resolution

    Returns:
        Path to the generated GIF file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    l1 = physical.l1
    l2 = physical.l2
    dt_frame = 1.0 / fps
    dt_sim = result.time[1] - result.time[0]
    skip = max(1, int(dt_frame / dt_sim))

    frames = []
    total_len = l1 + l2
    trail_x: list[float] = []
    trail_y: list[float] = []

    for i in range(0, len(result.states), skip):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)

        theta1 = result.states[i, 0]
        theta2 = result.states[i, 1]

        # Pivot at origin
        x0, y0 = 0.0, 0.0

        # Link 1 endpoint (theta1 from vertical downward)
        x1 = l1 * math.sin(theta1)
        y1 = -l1 * math.cos(theta1)

        # Link 2 endpoint
        x2 = x1 + l2 * math.sin(theta1 + theta2)
        y2 = y1 - l2 * math.cos(theta1 + theta2)

        trail_x.append(x2)
        trail_y.append(y2)

        # Draw trail (fading)
        max_trail = 200
        if len(trail_x) > max_trail:
            tx = trail_x[-max_trail:]
            ty = trail_y[-max_trail:]
        else:
            tx, ty = trail_x, trail_y
        if len(tx) > 1:
            ax.plot(tx, ty, "b-", alpha=0.15, linewidth=0.5)

        # Draw links
        ax.plot([x0, x1], [y0, y1], "k-", linewidth=4, solid_capstyle="round")
        ax.plot([x1, x2], [y1, y2], "C0-", linewidth=3, solid_capstyle="round")

        # Draw joints
        ax.plot(x0, y0, "ko", markersize=8, zorder=5)
        ax.plot(x1, y1, "C1o", markersize=6, zorder=5)
        ax.plot(x2, y2, "C3o", markersize=5, zorder=5)

        # Upright target line
        ax.plot([0, 0], [0, total_len], "r--", alpha=0.3, linewidth=1)

        # Info text
        t = result.time[i]
        E = result.energy[i]
        u = result.controls[min(i, len(result.controls) - 1)]
        err = abs(math.atan2(math.sin(theta1 - math.pi), math.cos(theta1 - math.pi)))
        mode = "LQR" if err < 0.3 and abs(result.states[i, 2]) < 1.0 else "Swing-Up"

        info = f"t={t:.2f}s | E={E:.1f}J | u={u:.1f}Nm | {mode}"
        ax.set_title(info, fontsize=10)

        margin = total_len * 1.3
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        # Render to image
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        img = Image.frombuffer("RGBA", (w, h), buf).convert("RGB")
        frames.append(img)
        plt.close(fig)

    # Save GIF
    path = f"{output_dir}/acrobot_animation.gif"
    if frames:
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
        )
    return path
