"""
Acrobot animation generator.

Optimized for speed and memory:
- Single figure/axes reused across all frames (no plt.subplots per frame)
- collections.deque for O(1) trail management (no list slicing copies)
- Pre-computed frame indices and mode array
- GIF saved without palette optimization for speed
"""

import math
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for speed
import matplotlib.pyplot as plt
from PIL import Image

from ..core.config import PhysicalParams
from ..core.types import SimulationResult


def generate_animation(
    result: SimulationResult,
    physical: PhysicalParams,
    output_dir: str = "output/gifs",
    fps: int = 30,
    dpi: int = 80,
) -> str:
    """Generate Acrobot animation GIF with optimized rendering.

    Uses single figure with cleared axes per frame instead of
    creating/destroying figure objects. Trail uses deque for
    O(1) append with automatic size limiting.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    l1, l2 = physical.l1, physical.l2
    total_len = l1 + l2
    margin = total_len * 1.3

    # Pre-compute frame indices
    dt_sim = result.time[1] - result.time[0]
    skip = max(1, int(1.0 / (fps * dt_sim)))
    frame_indices = list(range(0, len(result.states), skip))

    # Pre-compute mode for each frame (avoid per-frame atan2)
    mode_arr = np.array([
        abs(math.atan2(
            math.sin(result.states[i, 0] - math.pi),
            math.cos(result.states[i, 0] - math.pi)))
        for i in frame_indices
    ], dtype=np.float32)

    # Trail with O(1) append and automatic truncation
    max_trail = 200
    trail_x: deque[float] = deque(maxlen=max_trail)
    trail_y: deque[float] = deque(maxlen=max_trail)

    # Single figure reused for all frames
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)
    frames: list[Image.Image] = []

    for frame_idx, i in enumerate(frame_indices):
        ax.cla()  # Clear axes (reuse figure)

        theta1 = result.states[i, 0]
        theta2 = result.states[i, 1]

        # Link endpoints (scalar math, no array allocation)
        s1 = math.sin(theta1)
        c1 = math.cos(theta1)
        s12 = math.sin(theta1 + theta2)
        c12 = math.cos(theta1 + theta2)

        x1 = l1 * s1
        y1 = -l1 * c1
        x2 = x1 + l2 * s12
        y2 = y1 - l2 * c12

        trail_x.append(x2)
        trail_y.append(y2)

        # Draw trail (deque supports direct iteration)
        if len(trail_x) > 1:
            ax.plot(list(trail_x), list(trail_y), "b-", alpha=0.15, linewidth=0.5)

        # Draw links and joints
        ax.plot([0, x1], [0, y1], "k-", linewidth=4, solid_capstyle="round")
        ax.plot([x1, x2], [y1, y2], "C0-", linewidth=3, solid_capstyle="round")
        ax.plot(0, 0, "ko", markersize=8, zorder=5)
        ax.plot(x1, y1, "C1o", markersize=6, zorder=5)
        ax.plot(x2, y2, "C3o", markersize=5, zorder=5)

        # Upright target
        ax.plot([0, 0], [0, total_len], "r--", alpha=0.3, linewidth=1)

        # Info overlay (uses pre-computed mode)
        t = result.time[i]
        E = result.energy[i]
        u = result.controls[min(i, len(result.controls) - 1)]
        mode = "LQR" if mode_arr[frame_idx] < 0.3 else "Swing-Up"

        ax.set_title(f"t={t:.2f}s | E={E:.1f}J | u={u:.1f}Nm | {mode}",
                     fontsize=10)
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        # Render to image (reuses same canvas)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = Image.frombytes("RGBA", (w, h),
                              fig.canvas.buffer_rgba()).convert("RGB")
        frames.append(img)

    plt.close(fig)

    # Save GIF without optimization (faster save)
    path = f"{output_dir}/acrobot_animation.gif"
    if frames:
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=False,  # Skip palette quantization for speed
        )
    return path
