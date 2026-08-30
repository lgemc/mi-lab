from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .model import VizError
from .style import DIVERGING, PALETTE, SEQUENTIAL, set_style

"""
Charts of a comparison, which is a different subject from charts of a circuit.

A circuit chart draws one measurement over the model. These draw one
measurement over the *techniques*, and the axis that matters is almost always
the one nobody plots: what the technique cost. plot_cost_against_agreement is
the chart the rest exist to set up -- forward passes across, agreement with
the measurement being approximated up -- because a technique in the top left
is the one to use and a technique in the bottom right is one nobody should
run again.

The agreement matrices are drawn on a sequential scale rather than the
diverging one the head grids use. There is no meaningful zero in an overlap:
0 is "shared nothing" and it is the bottom of the range, not the middle of it.
The damage matrix does use the diverging scale, because a negative damage is a
real and different thing -- an ablation that made a task *better* -- and a
sequential map would draw it as almost nothing.

A common pipe could be: compare_techniques | plot_cost_against_agreement | save_figure
"""

def _square(matrix, labels: Sequence[str], name: str) -> np.ndarray:
    """Check that a chart was handed a square matrix matching its labels"""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise VizError(f"{name} must be a square matrix, got shape {values.shape}")
    if values.shape[0] != len(labels):
        raise VizError(f"{name} is {values.shape[0]} wide but {len(labels)} names were given for it")
    return values

def plot_agreement(comparison, which: str = "overlap", ax=None):
    """How much the techniques agree, as a technique-by-technique matrix

    'overlap' is intersection over union of the selected circuits and 'order'
    is rank correlation over every head. They are different questions and they
    come apart: two techniques can rank the whole tail identically and still
    disagree about the handful of heads that end up in the circuit, which is
    the only part anybody reads.
    """
    set_style()
    values = _square(comparison.matrix(which), comparison.methods, f"a {which} matrix")
    if ax is None:
        _, ax = plt.subplots(figsize=(0.9 * len(comparison.methods) + 3.2, 0.8 * len(comparison.methods) + 2.6))

    sns.heatmap(
        values, cmap=SEQUENTIAL, vmin=0, vmax=1, ax=ax, annot=True, fmt=".2f",
        xticklabels=comparison.methods, yticklabels=comparison.methods,
        cbar_kws={"label": "intersection over union" if which == "overlap" else "rank correlation"},
        linewidths=0.4, linecolor="white",
    )
    titles = {
        "overlap": f"Do the techniques pick the same {comparison.count} heads?",
        "order": "Do the techniques order every head the same way?",
    }
    ax.set_title(titles.get(which, which))
    ax.tick_params(axis="x", rotation=30)
    return ax

def plot_scores(comparison, ax=None):
    """Faithfulness, necessity and incompleteness for every technique's circuit

    All three at the same circuit size, because faithfulness climbs with the
    head count and a chart comparing techniques at different sizes compares
    sizes. Incompleteness is drawn as a cost rather than a score: it is the
    only one of the three where lower is better, and it is the one a faithful
    circuit fails.
    """
    set_style()
    results = [result for result in comparison.results if result.report is not None]
    if not results:
        raise VizError("no technique in this comparison was checked, so there are no scores to draw")
    if ax is None:
        _, ax = plt.subplots(figsize=(1.5 * len(results) + 3.4, 4.4))

    series = (
        ("faithfulness", [result.faithfulness for result in results], PALETTE["primary"]),
        ("necessity", [result.necessity for result in results], PALETTE["secondary"]),
        ("incompleteness", [result.incompleteness for result in results], PALETTE["highlight"]),
    )
    positions = np.arange(len(results))
    width = 0.26
    for index, (label, values, color) in enumerate(series):
        ax.bar(positions + (index - 1) * width, values, width, label=label, color=color)
    ax.axhline(1.0, color=PALETTE["grid"], linestyle=":", linewidth=1.5, label="clean behaviour")
    ax.axhline(0, color=PALETTE["grid"], linewidth=1)
    ax.set_xticks(positions, [result.method for result in results])
    ax.set_ylabel("recovery")
    ax.set_title(f"Every technique's {comparison.count} heads, checked the same three ways")
    ax.legend(frameon=False, ncol=2)
    return ax

def plot_cost_against_agreement(comparison, reference: str = "patching", ax=None):
    """What each technique cost against how well it agreed with the one it approximates

    The chart the rest exist to set up. Forward passes across on a log scale,
    because the techniques differ by two orders of magnitude and a linear axis
    draws four of them on top of each other; rank correlation with the
    reference up. Top left is a technique worth running on a model where the
    reference cannot be run at all.
    """
    set_style()
    found = comparison.by_method()
    if reference not in found:
        raise VizError(f"'{reference}' is not one of the techniques compared ({comparison.methods})")
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 5.0))

    for method, result in found.items():
        if method == reference:
            agreement = 1.0
        else:
            pair = comparison.order.get((method, reference), comparison.order.get((reference, method)))
            if pair is None:
                continue
            agreement = pair
        # a technique with no forward passes still has to be drawn somewhere on a log
        # axis, and half a pass is the honest place: cheaper than anything measurable
        passes = max(result.ranking.passes, 0.5)
        color = PALETTE["steered"] if method == reference else PALETTE["primary"]
        ax.scatter(passes, agreement, s=140, color=color, zorder=3)
        ax.annotate(method, (passes, agreement), textcoords="offset points", xytext=(9, 6), fontsize=10)

    ax.set_xscale("log")
    ax.axhline(0, color=PALETTE["grid"], linewidth=1)
    ax.set_xlabel("forward passes over the batch")
    ax.set_ylabel(f"rank correlation with {reference}")
    ax.set_title(f"What each technique cost, against how much of {reference} it recovered")
    return ax

def plot_consistency(consistency, ax=None):
    """How often each head turned up when the technique was run one example at a time

    The presence line is where the shared set is cut. The chance line is what
    two independent picks of this size would have shared anyway, and a
    distribution that never rises far above it means the batch-level circuit
    was an average of circuits with little in common.
    """
    set_style()
    frequency = consistency.frequency
    if not frequency:
        raise VizError("this consistency result has no per-example circuits to draw")
    if ax is None:
        _, ax = plt.subplots(figsize=(0.42 * len(frequency) + 3.6, 4.4))

    ordered = sorted(frequency.items(), key=lambda item: (-item[1], item[0]))
    labels = [f"L{layer}H{head}" for (layer, head), _ in ordered]
    shares = [share for _, share in ordered]
    colors = [PALETTE["primary"] if share >= consistency.presence else PALETTE["baseline"] for share in shares]

    ax.bar(range(len(shares)), shares, color=colors)
    ax.axhline(consistency.presence, color=PALETTE["steered"], linestyle="--", linewidth=1.5,
               label=f"shared at P={consistency.presence:g}")
    ax.axhline(consistency.chance, color=PALETTE["grid"], linestyle=":", linewidth=1.5,
               label=f"chance ({consistency.chance:.2f})")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_ylabel("share of per-example circuits containing it")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"{consistency.method} on {consistency.task}, one example at a time: reuse {consistency.reuse:.2f}"
    )
    ax.legend(frameon=False)
    return ax

def plot_specificity(specificity, ax=None):
    """What every task's circuit costs every task, as a damage matrix

    Rows are the task measured on and columns are the circuit ablated. The
    diagonal is the number a circuit paper reports; if the off-diagonal is
    close to it, the circuit is machinery the model uses for everything and
    the task label on it was never earned. The random column is the floor: any
    set of heads of that size costs the model something.
    """
    set_style()
    values = _square(specificity.matrix(), specificity.tasks, "a damage matrix")
    control = np.array([[specificity.control[task].damage] for task in specificity.tasks])
    grid = np.hstack([values, control])
    if ax is None:
        _, ax = plt.subplots(figsize=(1.15 * (len(specificity.tasks) + 1) + 3.4, 0.85 * len(specificity.tasks) + 2.8))

    limit = max(float(np.abs(grid).max()), 1e-9)
    sns.heatmap(
        grid, cmap=DIVERGING, center=0, vmin=-limit, vmax=limit, ax=ax, annot=True, fmt=".2f",
        xticklabels=[*specificity.tasks, "random"], yticklabels=specificity.tasks,
        cbar_kws={"label": "share of the clean logit difference lost"}, linewidths=0.4, linecolor="white",
    )
    ax.set_xlabel("circuit ablated")
    ax.set_ylabel("task measured on")
    ax.set_title("Ablating each task's circuit on every task")
    ax.tick_params(axis="x", rotation=30)
    return ax

def plot_technique_grid(ranking, title: Optional[str] = None, ax=None):
    """One technique's score for every head, laid out as layers down and heads across"""
    from .circuits import plot_head_grid

    return plot_head_grid(
        ranking.scores,
        title or f"{ranking.method}: every head scored in {ranking.units}",
        ranking.units,
        layers=ranking.layers,
        ax=ax,
    )
