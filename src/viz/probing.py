from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from ..core.metrics import best_threshold, roc_auc, roc_curve
from .model import VizError
from .style import PALETTE, annotate_best, class_colors, set_style

"""
Visualize what a probe found, and what it cost.

The layer sweep is the repo's headline result and currently prints as an ASCII
table. Drawn against depth rather than against layer index, two models put
their curves on the same axis and the claim "mid-layers carry the most probe
signal" becomes checkable at a glance instead of by comparing two tables with
different numbers of rows.

plot_method_sweep is the picture of the finding already written down in the
README: the logistic probe reads better than difference-of-means at every
depth and steers worse at all of them. Both curves on one axis is what makes
"AUC does not rank steering vectors" a chart rather than a claim.

plot_score_distribution is the one to look at when an AUC is suspiciously
round. AUC is rank-based, so it cannot tell a probe that separates the classes
by a wide margin from one that separates them by a hair, and those two have
very different futures under distribution shift.

plot_pareto is Module 2.3: five detection methods, cost on one axis and
accuracy on the other, with the frontier drawn. A method off the frontier is
dominated, and no amount of AUC argument brings it back.

A common pipe could be: sweep | plot_layer_sweep | plot_score_distribution
"""

def _flatten(activations: torch.Tensor) -> torch.Tensor:
    """Accept either a [batch, d_model] or a single-layer [batch, 1, d_model] capture"""
    if activations.dim() == 3 and activations.shape[1] == 1:
        return activations[:, 0]
    return activations

def plot_layer_sweep(reports: Sequence, metric: str = "auc", ax=None, label: Optional[str] = None, color: Optional[str] = None):
    """AUC (or accuracy) against depth, with the winning layer circled

    Plotted against depth fraction, not layer index, because that is the axis
    two different models share. The absolute height of the curve is a fact
    about this model and this dataset; the shape of it -- where the peak sits
    -- is the part expected to survive a model swap.
    """
    set_style()
    reports = list(reports)
    if not reports:
        raise VizError("a sweep chart needs at least one layer report")
    fracs = [report.frac for report in reports]
    values = [report.auc if metric == "auc" else report.metrics[metric] for report in reports]
    best = max(range(len(values)), key=lambda index: values[index])

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 5))

    color = color or PALETTE["primary"]
    ax.plot(fracs, values, marker="o", linewidth=2, color=color, label=label)
    annotate_best(ax, fracs[best], values[best],
                  f"layer {reports[best].layer}\n{metric} {values[best]:.3f}", color=color)

    ax.axhline(0.5, color=PALETTE["grid"], linestyle=":", linewidth=1)
    ax.text(0.005, 0.505, "chance", fontsize=8, color="0.45")
    ax.set_ylim(0.4, 1.03)
    ax.set_xlabel("Depth fraction")
    ax.set_ylabel(metric.upper() if metric == "auc" else metric)
    ax.set_title(f"Where the signal lives: probe {metric} by depth")
    if label:
        ax.legend()
    return ax

def plot_method_sweep(sweeps: Dict[str, Sequence], metric: str = "auc", ax=None):
    """Several sweeps on one axis, one line per probe method or per model

    Keys become the legend, so this draws logistic against difference-of-means
    on one model, or one method across two models, without needing to know
    which comparison it is being asked for.
    """
    set_style()
    if not sweeps:
        raise VizError("a method comparison needs at least one sweep")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.5))

    cycle = [PALETTE["primary"], PALETTE["steered"], PALETTE["secondary"], PALETTE["accent"], PALETTE["highlight"]]
    for index, (name, reports) in enumerate(sweeps.items()):
        plot_layer_sweep(reports, metric=metric, ax=ax, label=name, color=cycle[index % len(cycle)])
    ax.set_title(f"Probe {metric} by depth, {len(sweeps)} methods compared")
    ax.legend(title="")
    return ax

def plot_roc(probe, activations: torch.Tensor, labels: Sequence[int], ax=None):
    """The ROC curve whose area the sweep reports as one number

    Worth drawing when an AUC is near 1.0: a curve that hugs the top-left
    corner and one that reaches it in a single step both score the same, and
    the second one is a probe with almost no thresholds to choose between.
    """
    set_style()
    scores = probe.score(_flatten(activations))
    false_positives, true_positives = roc_curve(scores, labels)
    area = roc_auc(scores, labels)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.plot(false_positives, true_positives, linewidth=2.2, color=PALETTE["primary"],
            label=f"{probe.method} (AUC {area:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=PALETTE["grid"], label="chance")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC at layer {probe.layer} of {probe.model_id}")
    ax.legend(loc="lower right")
    return ax

def plot_score_distribution(
    probe,
    activations: torch.Tensor,
    labels: Sequence[int],
    label_names: Optional[Sequence[str]] = None,
    ax=None,
):
    """Overlaid histograms of the probe's score for each class

    The margin between the two humps is what AUC throws away. A wide gap is a
    probe that will survive a threshold being set on different data; two humps
    that touch is a probe whose threshold is a coin flip on anything it has
    not seen.
    """
    set_style()
    scores = probe.score(_flatten(activations)).tolist()
    names = tuple(label_names or ("negative", "positive"))
    colors = class_colors(names)

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 4.8))

    for value, name in enumerate(names):
        selected = [score for score, label in zip(scores, labels, strict=True) if label == value]
        sns.histplot(selected, bins=25, ax=ax, color=colors[name], label=name, alpha=0.55, element="step")

    threshold, accuracy_at = best_threshold(scores, labels)
    ax.axvline(0.0, color=PALETTE["grid"], linestyle=":", linewidth=1.2, label="decision boundary")
    ax.axvline(threshold, color=PALETTE["steered"], linestyle="--", linewidth=1.4,
               label=f"best threshold {threshold:.2f} (acc {accuracy_at:.3f})")
    ax.set_xlabel("Probe score (signed distance from the boundary)")
    ax.set_ylabel("Examples")
    ax.set_title(f"Score separation at layer {probe.layer} ({probe.method})")
    ax.legend(fontsize=8)
    return ax

def plot_weight_spectrum(probe, k: int = 20):
    """How concentrated the probe's direction is, and which dimensions carry it

    Left panel: the sorted magnitude of every component of the direction in
    activation space, on a log axis. A cliff means a handful of residual
    stream dimensions carry the property; a straight line means it is spread
    over the whole stream. Right panel: the k largest, signed.

    The direction is used rather than the raw weight, because the weight lives
    in standardized coordinates and its large components are as much a fact
    about that layer's activation scale as about the probe.

    Returns a Figure: there are two panels.
    """
    set_style()
    direction = probe.direction.float()
    magnitudes, order = direction.abs().sort(descending=True)
    take = min(k, direction.numel())

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.3, 1]})

    left.plot(range(1, magnitudes.numel() + 1), magnitudes.tolist(), color=PALETTE["primary"], linewidth=1.8)
    left.set_yscale("log")
    left.set_xlabel("Dimension, sorted by magnitude")
    left.set_ylabel("|component|")
    share = float(magnitudes[:take].sum() / magnitudes.sum().clamp_min(1e-12))
    left.set_title(f"Weight spectrum: top {take} carry {share:.0%} of the mass")

    top_dimensions = order[:take].tolist()
    values = direction[order[:take]].tolist()
    colors = [PALETTE["positive"] if value > 0 else PALETTE["negative"] for value in values]
    right.barh([str(dimension) for dimension in top_dimensions], values, color=colors)
    right.invert_yaxis()
    right.axvline(0, color=PALETTE["grid"], linewidth=1)
    right.set_xlabel("Signed component")
    right.set_ylabel("Residual stream dimension")
    right.set_title(f"Top {take} dimensions")

    figure.suptitle(f"Probe direction at layer {probe.layer} of {probe.model_id} ({probe.method})")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure

def pareto_frontier(points: Sequence[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    """The (name, cost, score) entries no other entry beats on both axes

    Cheaper is better and higher scoring is better, so an entry survives only
    if nothing else is at once no more expensive and no less accurate.
    """
    frontier = []
    for entry in sorted(points, key=lambda point: (point[1], -point[2])):
        if not frontier or entry[2] > frontier[-1][2]:
            frontier.append(entry)
    return frontier

def plot_pareto(points: Sequence[Tuple[str, float, float]], score_name: str = "AUC", ax=None):
    """Cost against accuracy for several methods, with the frontier drawn

    Each point is (name, milliseconds per item, score). Anything below and to
    the right of the frontier is dominated: something else is both cheaper and
    better, and the only honest thing to do with it is stop proposing it.
    """
    set_style()
    points = list(points)
    if not points:
        raise VizError("a Pareto chart needs at least one method")
    frontier = pareto_frontier(points)
    on_frontier = {name for name, _, _ in frontier}

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 5.5))

    for name, cost, score in points:
        is_best = name in on_frontier
        ax.scatter([cost], [score], s=150 if is_best else 90,
                   color=PALETTE["primary"] if is_best else PALETTE["baseline"],
                   edgecolors="white", linewidths=1.4, zorder=3)
        ax.annotate(name, (cost, score), textcoords="offset points", xytext=(9, 6), fontsize=9,
                    weight="bold" if is_best else "normal",
                    color="0.15" if is_best else "0.45")

    ax.plot([point[1] for point in frontier], [point[2] for point in frontier],
            linestyle="--", linewidth=1.6, color=PALETTE["primary"], alpha=0.65, zorder=2,
            label="Pareto frontier")

    ax.set_xscale("log")
    ax.set_xlabel("Cost (ms per item, log scale)")
    ax.set_ylabel(score_name)
    ax.set_title(f"Cost against {score_name}: {len(points) - len(frontier)} of {len(points)} methods dominated")
    ax.legend(loc="lower right")
    return ax
