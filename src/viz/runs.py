from typing import List, Sequence

import matplotlib.pyplot as plt
import seaborn as sns

from .model import VizError
from .style import PALETTE, set_style

"""
Visualize what a pile of runs adds up to, reading only run.json.

Nothing here imports torch, for the same reason core.run does not: a run is
plain data, and looking at a hundred of them should not need the library that
produced them.

plot_metric_across_runs is the scaling contract as a chart. The table in the
course index lists what should hold across a model swap and what should not;
running one experiment against two configs and putting the metric side by side
is how that table gets checked instead of quoted.

plot_sweep_from_run reads the per-layer AUCs a probe_sweep recorded, so a
finished run redraws its own depth curve without recomputing anything.

A common pipe could be: find_runs | plot_metric_across_runs
"""

def _completed(runs: Sequence) -> List:
    """Keep the runs that finished, since a failed run has no metric to plot"""
    return [run for run in runs if run.status == "completed"]

def plot_metric_across_runs(runs: Sequence, metric: str = "best_auc", group_by: str = "experiment", ax=None):
    """Bar chart of one metric across runs, newest last

    group_by names the field that labels each bar: 'experiment' compares
    experiments, and any dotted path into the run's recorded params compares
    what was swapped -- 'model.config' is the model-swap comparison.
    """
    set_style()
    runs = _completed(runs)
    if not runs:
        raise VizError("no completed runs to plot")

    labels, values = [], []
    for run in reversed(runs):
        if metric not in run.metrics:
            continue
        labels.append(_label(run, group_by))
        values.append(run.metrics[metric])
    if not values:
        raise VizError(
            f"none of the {len(runs)} completed runs recorded '{metric}'; "
            f"metrics seen are {sorted({key for run in runs for key in run.metrics})}"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels) + 2), 5))

    sns.barplot(x=labels, y=values, ax=ax, hue=labels, legend=False, palette="crest")
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_xlabel(group_by)
    ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"{metric} across {len(values)} runs")
    return ax

def _label(run, group_by: str) -> str:
    """Resolve a dotted path into a run's params, falling back to its fields"""
    if hasattr(run, group_by):
        return str(getattr(run, group_by))
    value = run.params
    for key in group_by.split("."):
        if not isinstance(value, dict) or key not in value:
            return run.run_id[:15]
        value = value[key]
    return str(value)

def plot_sweep_from_run(run, ax=None):
    """Redraw a probe_sweep's depth curve from the metrics it recorded

    A completed sweep records auc_layer_N for every layer it tried, so the
    chart comes back without the forward passes that produced it.
    """
    set_style()
    layers = sorted(
        (int(key.rsplit("_", 1)[1]), value)
        for key, value in run.metrics.items()
        if key.startswith("auc_layer_")
    )
    if not layers:
        raise VizError(
            f"run {run.run_id} recorded no per-layer AUCs; only a probe_sweep does, "
            f"and this run is a {run.kind}"
        )

    indices = [layer for layer, _ in layers]
    values = [value for _, value in layers]

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 5))

    ax.plot(indices, values, marker="o", linewidth=2, color=PALETTE["primary"])
    best = run.metrics.get("best_layer")
    if best is not None and int(best) in indices:
        position = indices.index(int(best))
        ax.scatter([indices[position]], [values[position]], s=140, facecolors="none",
                   edgecolors=PALETTE["steered"], linewidths=2, zorder=5)
    ax.axhline(0.5, color=PALETTE["grid"], linestyle=":", linewidth=1)
    ax.set_ylim(0.4, 1.03)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC")
    ax.set_title(f"{run.experiment}: probe AUC by layer")
    return ax
