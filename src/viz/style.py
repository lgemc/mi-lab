from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import seaborn as sns

"""
Shared plotting style, a palette named by meaning, and the save/show helpers
every chart in this package goes through, so figures look like one family and
land on disk the same way.

The palette is keyed by what a series *is* rather than by what colour it
happens to be. A chart that says PALETTE["baseline"] stays correct when the
theme changes; one that says "steelblue" quietly stops meaning anything the
moment a second chart picks a different blue for the same series. The two
class colours are the same two everywhere, so a positive class is the same
green in the dataset chart and in the probe's score histogram.

Nothing here selects a matplotlib backend. The CLI writes files and picks Agg
itself; a notebook importing these functions keeps whatever backend it had,
which forcing Agg here would take away.

A common pipe could be: set_style | plot_* | save_figure | show_figure
"""

PALETTE = {
    "positive": "#2a9d8f",
    "negative": "#e76f51",
    "primary": "#4c72b0",
    "secondary": "#55a868",
    "accent": "#8172b3",
    "baseline": "#8d99ae",
    "steered": "#c44e52",
    "highlight": "#dd8452",
    "grid": "#b0b7c3",
}

# Diverging for anything centred on a meaningful zero (drift, patch effects,
# weights); sequential for magnitudes that have no midpoint (norms, counts).
DIVERGING = "RdBu_r"
SEQUENTIAL = "magma"

CLASS_COLORS = (PALETTE["negative"], PALETTE["positive"])

def set_style():
    """Apply the seaborn theme used by every plot in this package"""
    sns.set_theme(style="whitegrid", context="notebook")

def class_colors(label_names: Optional[Sequence[str]] = None) -> dict:
    """Map class names to their fixed colours, defaulting to negative/positive

    Datasets may carry their own label_names, so this takes them rather than
    assuming the two classes are called anything in particular.
    """
    names = tuple(label_names or ("negative", "positive"))
    return dict(zip(names, CLASS_COLORS, strict=True))

def figure_of(target):
    """The Figure behind an Axes, or the Figure itself

    Plot functions return an Axes when there is one panel and a Figure when
    there is a grid, and every caller wants to save either without caring.
    """
    return target if isinstance(target, plt.Figure) else target.figure

def save_figure(target, path):
    """Save the figure containing the given axes to disk, creating parent dirs"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = figure_of(target)
    figure.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(figure)
    return path

def show_figure(path):
    """Render a saved chart inline in the terminal

    Uses term-image, which auto-picks the best available protocol for the
    current terminal (Kitty, iTerm2, Sixel) and falls back to Unicode blocks
    everywhere else, so this works even over plain/unsupported terminals.
    """
    from term_image.image import from_file

    from_file(str(path)).draw()

def annotate_best(ax, x, y, text: str, color: Optional[str] = None):
    """Mark the winning point on a sweep, since which one won is the whole result"""
    ax.scatter([x], [y], s=140, facecolors="none", edgecolors=color or PALETTE["steered"], linewidths=2, zorder=5)
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(8, 8), fontsize=9,
                color=color or PALETTE["steered"], weight="bold")
    return ax
