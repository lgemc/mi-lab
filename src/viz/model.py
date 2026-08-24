from typing import List, Optional, Sequence

import matplotlib.pyplot as plt

from .style import PALETTE, SEQUENTIAL, set_style

"""
Visualize what a config resolves to, which is the one thing in this repo with
no numeric output worth reading as a table.

The depth ruler is the chart this framework exists to make. Every model gets a
lane running from the embedding to the unembedding, marked with its own layer
boundaries, and every requested depth fraction is one vertical line crossing
all of them. The line is straight; the integers under it are not. Depth 0.65
is layer 8 on GPT-2 small, layer 16 on GPT-2 medium and layer 42 on a 64-layer
model, and reading those three numbers off one vertical line is the argument
for addressing layers by fraction in about two seconds.

Both functions take resolved ModelConfigs and read n_layers off them. Nothing
here loads a checkpoint: resolution is the caller's job, so a notebook holding
adapters and a CLI that just loaded four of them use the same function.

A common pipe could be: load_config | with_sizes | plot_depth_ruler
"""

class VizError(ValueError):
    """Raised when a chart is asked to draw something the data cannot support"""

def _require_resolved(configs: Sequence) -> List:
    """Reject configs whose checkpoint sizes have not been filled in yet

    An unresolved config has no n_layers, so a fraction cannot be turned into
    an index and the lane would have nothing to draw.
    """
    configs = list(configs)
    if not configs:
        raise VizError("the depth ruler needs at least one config")
    unresolved = [cfg.id for cfg in configs if not cfg.is_resolved]
    if unresolved:
        raise VizError(
            f"configs {unresolved} have no n_layers yet; load an adapter for them first, "
            "or write n_layers into the config file"
        )
    return configs

def plot_depth_ruler(configs: Sequence, fracs: Optional[Sequence[float]] = None, ax=None):
    """One lane per model, with each depth fraction crossing all of them

    The absolute layer index printed at every crossing is what changes across
    models; the fraction is what does not. A fraction that lands on the same
    relative place but a different integer on each lane is the framework
    working.
    """
    set_style()
    configs = _require_resolved(configs)
    fracs = list(fracs) if fracs is not None else [0.0, 0.25, 0.5, 0.65, 0.85, 1.0]

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 0.95 * len(configs) + 2))

    for frac in fracs:
        is_probe_depth = any(abs(frac - cfg.probe_layer_frac) < 1e-9 for cfg in configs)
        ax.axvline(
            frac,
            color=PALETTE["steered"] if is_probe_depth else PALETTE["grid"],
            linewidth=1.6 if is_probe_depth else 0.9,
            linestyle="-" if is_probe_depth else "--",
            zorder=1,
        )
        ax.text(frac, len(configs) - 0.30, f"{frac:g}", ha="center", va="bottom", fontsize=9,
                zorder=6, color=PALETTE["steered"] if is_probe_depth else "0.35",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.85})

    for row, cfg in enumerate(reversed(configs)):
        y = row
        ax.plot([0, 1], [y, y], color=PALETTE["primary"], linewidth=2.5, solid_capstyle="round", zorder=2)
        # every layer boundary of this model, so the lane shows its own granularity
        for index in range(cfg.n_layers):
            ax.plot([index / cfg.n_layers], [y], marker="|", markersize=9,
                    color=PALETTE["primary"], alpha=0.45, zorder=3)
        for frac in fracs:
            resolved = cfg.layer(frac)
            is_probe_depth = abs(frac - cfg.probe_layer_frac) < 1e-9
            ax.scatter([frac], [y], s=110, zorder=4,
                       color=PALETTE["steered"] if is_probe_depth else PALETTE["accent"],
                       edgecolors="white", linewidths=1.2)
            # zorder and a matching backdrop keep the index readable where the
            # highlighted fraction's line runs straight through the label
            ax.annotate(f"{resolved}", (frac, y), textcoords="offset points", xytext=(0, -18),
                        ha="center", fontsize=9, weight="bold", zorder=6,
                        color=PALETTE["steered"] if is_probe_depth else "0.25",
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.85})

    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([f"{cfg.id}\n{cfg.n_layers} layers, d={cfg.d_model}" for cfg in reversed(configs)])
    ax.set_ylim(-0.7, len(configs) - 0.1)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xlabel("Depth fraction")
    ax.set_title("Depth ruler: one fraction, one line, a different layer index per model")
    ax.grid(False)
    return ax

def plot_shape(configs: Sequence, ax=None):
    """Scatter of residual stream width against depth, one point per model

    Models do not scale by getting deeper or wider alone, and where a
    checkpoint sits in this plane is what decides whether a result about
    "mid-layers" is a claim about six layers or sixty-four.
    """
    set_style()
    configs = _require_resolved(configs)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5.5))

    widths = [cfg.d_model for cfg in configs]
    depths = [cfg.n_layers for cfg in configs]
    # area of the residual stream is what a capture actually costs to hold
    sizes = [0.0006 * width * depth for width, depth in zip(widths, depths, strict=True)]

    scatter = ax.scatter(widths, depths, s=sizes, c=[width * depth for width, depth in zip(widths, depths, strict=True)],
                         cmap=SEQUENTIAL, alpha=0.75, edgecolors="white", linewidths=1.5)
    for cfg in configs:
        ax.annotate(cfg.id, (cfg.d_model, cfg.n_layers), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("d_model (residual stream width)")
    ax.set_ylabel("n_layers (depth)")
    ax.set_title("Model shape: width against depth")
    ax.figure.colorbar(scatter, ax=ax, label="n_layers x d_model")
    return ax
