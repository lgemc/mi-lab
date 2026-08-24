from typing import Dict, Sequence

import matplotlib.pyplot as plt

from .model import VizError
from .style import PALETTE, set_style

"""
Visualize a strength sweep, which is the only honest shape for a steering
result.

plot_strength_sweep puts effect and fluency on the same x axis with separate y
axes, because the ceiling is not a value either curve reaches on its own -- it
is where one is still climbing and the other has started to fall. A single
generation at one strength cannot show that, and picking the strength that
made the nicest sentence is how a steering demo becomes a steering claim it
cannot support.

plot_control_comparison is Module 4.4 drawn: the real direction against a
random vector of the same norm. Both move the model. If the gap between them
is small, the intervention was size and not direction, and the honest report
says so.

A common pipe could be: strength_sweep | plot_strength_sweep
"""

def _series(points: Sequence, attribute: str):
    """Pull one measured attribute off a list of SteeringPoints"""
    return [getattr(point, attribute) for point in points]

def plot_strength_sweep(points: Sequence, ax=None, effect_name: str = "probe score"):
    """Effect and fluency against steering strength, on twin axes

    Read it as two questions at once: is the behaviour arriving, and is the
    model still writing sentences. The useful strength is the largest one
    where the answer to both is yes.
    """
    set_style()
    points = list(points)
    if not points:
        raise VizError("a strength sweep chart needs at least one point")
    strengths = _series(points, "strength")
    effects = _series(points, "effect")
    fluencies = _series(points, "fluency")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.2))

    ax.plot(strengths, effects, marker="o", linewidth=2.2, color=PALETTE["steered"], label=effect_name)
    ax.set_xlabel("Steering strength (mean activation norms)")
    ax.set_ylabel(effect_name, color=PALETTE["steered"])
    ax.tick_params(axis="y", labelcolor=PALETTE["steered"])

    twin = ax.twinx()
    twin.plot(strengths, fluencies, marker="s", linewidth=2, linestyle="--",
              color=PALETTE["primary"], label="fluency (distinct-word share)")
    twin.set_ylabel("Fluency", color=PALETTE["primary"])
    twin.tick_params(axis="y", labelcolor=PALETTE["primary"])
    twin.set_ylim(0, 1.05)
    twin.grid(False)

    # the ceiling: the largest strength before fluency turns down for good
    ceiling = None
    for index in range(1, len(fluencies)):
        if fluencies[index] < fluencies[index - 1] - 0.05:
            ceiling = strengths[index - 1]
            break
    if ceiling is not None:
        ax.axvline(ceiling, color=PALETTE["grid"], linestyle=":", linewidth=1.5)
        ax.text(ceiling, ax.get_ylim()[1], f" ceiling ~{ceiling:g}", va="top", fontsize=9, color="0.35")

    handles = ax.get_lines() + twin.get_lines()
    ax.legend(handles, [line.get_label() for line in handles], loc="lower left", fontsize=9)
    ax.set_title("Steering strength: effect against the damage it does")
    return ax

def plot_control_comparison(sweeps: Dict[str, Sequence], attribute: str = "effect", ax=None):
    """One line per direction, so a real one can be read against its random control

    Keys become the legend. The gap between the real direction and the control
    is the entire result; the height of the real curve on its own is not.
    """
    set_style()
    if not sweeps:
        raise VizError("a control comparison needs at least one sweep")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.2))

    cycle = [PALETTE["steered"], PALETTE["baseline"], PALETTE["accent"], PALETTE["secondary"]]
    for index, (name, points) in enumerate(sweeps.items()):
        points = list(points)
        style = "--" if "control" in name.lower() or "random" in name.lower() else "-"
        ax.plot(_series(points, "strength"), _series(points, attribute), marker="o",
                linewidth=2.2, linestyle=style, color=cycle[index % len(cycle)], label=name)

    ax.set_xlabel("Steering strength (mean activation norms)")
    ax.set_ylabel(attribute)
    ax.set_title(f"{attribute.capitalize()} against strength: direction versus control")
    ax.legend()
    return ax
