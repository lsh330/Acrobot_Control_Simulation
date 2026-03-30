"""
Publication-quality matplotlib style configuration.

Sets consistent styling for all plots matching SCI journal standards.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_publication_style() -> None:
    """Apply publication-quality matplotlib style."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (10, 8),
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
