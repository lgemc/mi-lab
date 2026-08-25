from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from .model import VizError
from .style import DIVERGING, PALETTE, SEQUENTIAL, annotate_best, set_style

"""
Visualize a circuit study, which is two claims that have to be drawn against
each other rather than one after the other.

Every layer-by-head chart here uses the same diverging scale centred on zero,
because in this subject the sign is the finding: a head that pushes the answer
away is not a weak head, it is a different kind of head, and a sequential
colormap would draw it as "nearly nothing".

plot_attribution_against_effect is the chart the rest exist to set up. Direct
attribution on one axis, causal patching on the other, and the heads that sit
off the diagonal are the whole point -- the ones that write the answer and get
overruled, and the ones that never touch the answer but decide who does.

A common pipe could be: direct_logit_attribution | patch_heads | plot_attribution_against_effect
"""

def _grid(values: torch.Tensor, name: str) -> torch.Tensor:
    """Check that a chart was handed a layer-by-head matrix"""
    values = torch.as_tensor(values).float()
    if values.dim() != 2:
        raise VizError(f"{name} must be a [layer, head] matrix, got shape {tuple(values.shape)}")
    return values

def _symmetric(values: torch.Tensor) -> float:
    """The limit that puts zero in the middle of the colour scale"""
    return max(float(values.abs().max()), 1e-9)

def plot_head_grid(values: torch.Tensor, title: str, legend: str, layers: Optional[Sequence[int]] = None, ax=None):
    """One number per attention head, laid out as layers down and heads across

    The scale is symmetric around zero on purpose: red and blue mean "towards
    the answer" and "away from it", and they mean the same thing in every
    chart in this package, so an attribution grid and a patching grid can be
    read side by side.
    """
    set_style()
    values = _grid(values, "a head grid")
    if ax is None:
        _, ax = plt.subplots(figsize=(0.62 * values.shape[1] + 3, 0.42 * values.shape[0] + 2.4))

    rows = list(layers) if layers else list(range(values.shape[0]))
    limit = _symmetric(values)
    sns.heatmap(
        values.numpy(), cmap=DIVERGING, center=0, vmin=-limit, vmax=limit, ax=ax,
        xticklabels=range(values.shape[1]), yticklabels=rows,
        cbar_kws={"label": legend}, linewidths=0.4, linecolor="white",
    )
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    return ax

def plot_attribution(attribution, ax=None):
    """What each head wrote towards the answer along the direct path to the unembedding"""
    return plot_head_grid(
        attribution.heads,
        f"Direct logit attribution: {attribution.measured:+.2f} logits of answer, split by head",
        "logit difference contributed",
        ax=ax,
    )

def plot_head_effects(effects, ax=None):
    """How much of the clean behaviour each head restores on its own, patched from the clean run"""
    return plot_head_grid(
        effects.effects,
        f"Head patching: share of the {effects.baselines.span:+.2f} logit span each head recovers",
        "recovery (0 corrupted, 1 clean)",
        layers=effects.layers,
        ax=ax,
    )

def plot_top_heads(ranked: Sequence, count: int = 12, legend: str = "effect", ax=None):
    """The strongest heads as a bar chart, signed, largest first

    A heatmap says where; this says how much, and it is the one to put in
    front of somebody who has not been staring at a twelve-by-twelve grid.
    """
    set_style()
    ranked = list(ranked)[:count]
    if not ranked:
        raise VizError("there are no heads to rank")

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 0.42 * len(ranked) + 1.8))

    labels = [f"L{layer}H{head}" for (layer, head), _ in ranked]
    values = [value for _, value in ranked]
    colors = [PALETTE["positive"] if value >= 0 else PALETTE["negative"] for value in values]
    ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(values)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color=PALETTE["grid"], linewidth=1)
    ax.set_xlabel(legend)
    ax.set_title(f"The {len(ranked)} heads that move the answer most")
    return ax

def plot_patch_grid(grid, ax=None):
    """Recovery from restoring the clean residual stream at one layer and position

    Read it as a story rather than a table. The bright cells start where the
    sentence repeats a name and move to the last position partway up the
    model: that jump is the moment the answer stops being a fact about a token
    earlier in the sentence and becomes a fact about what comes next.
    """
    set_style()
    values = _grid(grid.effects, "a patching grid")
    if len(grid.tokens) != values.shape[1]:
        raise VizError(f"{values.shape[1]} positions in the grid but {len(grid.tokens)} token labels")

    if ax is None:
        _, ax = plt.subplots(figsize=(0.58 * values.shape[1] + 4, 0.4 * values.shape[0] + 2.6))

    limit = _symmetric(values)
    named = {position: name for name, position in grid.landmarks.items()}
    labels = [
        f"{token.strip() or '␣'}\n[{named[index]}]" if index in named else token.strip() or "␣"
        for index, token in enumerate(grid.tokens)
    ]
    sns.heatmap(
        values.numpy(), cmap=DIVERGING, center=0, vmin=-limit, vmax=limit, ax=ax,
        xticklabels=labels, yticklabels=grid.layers or list(range(values.shape[0])),
        cbar_kws={"label": "recovery (0 corrupted, 1 clean)"}, linewidths=0.4, linecolor="white",
    )
    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer")
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.set_title("Restoring the clean residual stream, one layer and position at a time")
    return ax

def plot_roles(roles, threshold: float = 0.3, axes=None):
    """One panel per movement, showing how much attention each head spends on it

    Four small heatmaps rather than one classification chart, because a head
    doing two jobs at once is common and a single label per head hides it.
    """
    set_style()
    weights = torch.as_tensor(roles.weights).float()
    if weights.dim() != 3:
        raise VizError(f"head roles must be [layer, head, role], got shape {tuple(weights.shape)}")

    names = list(roles.roles)
    # every panel shares one scale so they can be compared, and it is the observed
    # maximum rather than 1.0: attention that peaks near a half draws as black
    # against a full probability scale, which hides the finding to make a point
    ceiling = max(float(weights.max()), threshold)
    if axes is None:
        figure, axes = plt.subplots(1, len(names), figsize=(4.1 * len(names), 0.34 * weights.shape[0] + 3),
                                    sharey=True)
    else:
        figure = axes[0].figure

    for index, (name, ax) in enumerate(zip(names, axes, strict=True)):
        sns.heatmap(
            weights[:, :, index].numpy(), cmap=SEQUENTIAL, vmin=0, vmax=ceiling, ax=ax,
            xticklabels=range(weights.shape[1]), yticklabels=range(weights.shape[0]),
            cbar=index == len(names) - 1, cbar_kws={"label": f"attention weight (0 to {ceiling:.2f})"},
            linewidths=0.3, linecolor="white",
        )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Head")
        ax.tick_params(axis="x", rotation=0)
    axes[0].set_ylabel("Layer")
    figure.suptitle(f"Where each head looks, by role (named above {threshold:g})", y=1.02)
    return figure

def plot_attribution_against_effect(attribution, effects, labels: Optional[Dict] = None, ax=None):
    """Direct attribution against causal effect, one point per head

    The diagonal is where the two agree. Everything interesting is off it: a
    head high on attribution and flat on patching wrote the answer and had it
    taken back, and a head flat on attribution and high on patching never
    touched the answer -- it decided which other head would.
    """
    set_style()
    caused = _grid(effects.effects, "a head effect grid")
    # attribution always covers the whole model; head patching may have swept only part of it
    rows = list(effects.layers) or list(range(caused.shape[0]))
    written = _grid(attribution.heads, "an attribution grid")[rows]
    if written.shape != caused.shape:
        raise VizError(f"attribution is {tuple(written.shape)} but head effects are {tuple(caused.shape)}")

    if ax is None:
        _, ax = plt.subplots(figsize=(8.2, 6))

    layers = torch.tensor(rows).repeat_interleave(written.shape[1])
    points = ax.scatter(
        written.flatten(), caused.flatten(), c=layers, cmap="viridis", s=52,
        edgecolors="white", linewidths=0.6, zorder=3,
    )
    ax.figure.colorbar(points, ax=ax, label="Layer")
    ax.axhline(0, color=PALETTE["grid"], linewidth=1)
    ax.axvline(0, color=PALETTE["grid"], linewidth=1)

    # labels are staggered because the interesting heads cluster, and two of them
    # overprinted says less than either would have said alone
    offsets = ((9, 9), (9, -16), (-52, 9), (-52, -16))
    for index, ((layer, head), _) in enumerate((labels or {}).items()):
        if layer not in rows:
            continue
        row = rows.index(layer)
        annotate_best(ax, float(written[row, head]), float(caused[row, head]), f"L{layer}H{head}",
                      offset=offsets[index % len(offsets)])

    ax.set_xlabel("Direct logit attribution (logits written towards the answer)")
    ax.set_ylabel("Head patching (share of the span recovered)")
    ax.set_title("What a head writes against what a head causes")
    return ax

def plot_circuit_growth(circuit, ax=None):
    """Recovery after each head the greedy search added, in the order it added them

    The shape is the result. A curve that jumps and then flattens found a
    circuit; one that climbs evenly at every step found a model doing the task
    everywhere, and calling the first eight heads a circuit would be a choice
    about where to stop rather than a finding.
    """
    set_style()
    scores = list(circuit.scores)
    if not scores:
        raise VizError("this circuit was never scored as it grew, so there is nothing to plot")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.8))

    steps = range(1, len(scores) + 1)
    ax.plot(steps, scores, marker="o", linewidth=2.2, color=PALETTE["primary"])
    ax.axhline(circuit.threshold, color=PALETTE["steered"], linestyle="--", linewidth=1.5,
               label=f"threshold {circuit.threshold:g}")
    ax.axhline(1.0, color=PALETTE["grid"], linestyle=":", linewidth=1.5, label="clean behaviour")
    ax.set_xticks(list(steps), [f"L{layer}H{head}" for layer, head in circuit.heads], rotation=45, ha="right")
    ax.set_xlabel("Heads restored, in the order the search added them")
    ax.set_ylabel("Recovery")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("How much of the answer comes back as the circuit grows")
    return ax

def plot_verification(report, ax=None):
    """Faithfulness, necessity, and what each head costs the circuit when dropped

    Minimality is drawn on the same axis rather than in its own chart, because
    a bar near zero next to a faithfulness of 0.9 is the sentence "this head is
    a passenger", and splitting the two makes it a comparison nobody makes.
    """
    set_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8.4, 5))

    labels = ["faithfulness", "necessity"] + [f"L{layer}H{head}" for layer, head in report.minimality]
    values = [report.faithfulness, report.necessity, *report.minimality.values()]
    colors = [PALETTE["primary"], PALETTE["accent"]] + [
        PALETTE["positive"] if drop >= 0.05 else PALETTE["baseline"] for drop in report.minimality.values()
    ]
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)), labels, rotation=45, ha="right")
    ax.axhline(0, color=PALETTE["grid"], linewidth=1)
    ax.set_ylabel("Recovery, or the recovery lost by dropping this head")
    ax.set_title(f"Is the circuit enough, is it needed, is any of it spare ({len(report.circuit)} heads)")
    return ax
