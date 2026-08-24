import math
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from .model import VizError
from .style import DIVERGING, PALETTE, SEQUENTIAL, class_colors, set_style

"""
Visualize what came out of adapter.capture, which is a [batch, layer, d_model]
tensor and otherwise unreadable.

Two of these are sanity checks and the rest are findings.

plot_layer_norms is the check the CLI already prints as text: a layer whose
activations are zero, or indistinguishable from an early layer's, means the
hook is not where you think it is. Residual stream norms grow roughly
monotonically with depth in a healthy capture, and a flat profile is a bug.

plot_layer_similarity is the second check and a fact at once. Adjacent layers
of a residual stream are nearly parallel because each block adds to the stream
rather than replacing it, so the matrix should be dominated by its diagonal
band. A block-diagonal structure with a visible seam is the model changing what
it is doing at that depth.

plot_layer_pca is where the probe result comes from. If two classes separate
visibly in two dimensions at some layer, a linear probe will find them there;
if the sweep says layer 8 wins, this is the picture of why.

plot_drift is Module 7.2: the same prompts through two models -- two
quantizations, two checkpoints -- compared layer by layer. Cosine near 1.0 at
every depth means the internals survived; a knee is where they stopped.

A common pipe could be: capture | plot_layer_norms | plot_layer_pca
"""

def _as_batch_layer_model(activations: torch.Tensor) -> torch.Tensor:
    """Insist on a [batch, layer, d_model] capture, naming the shape if it is not"""
    if activations.dim() != 3:
        raise VizError(
            f"expected a [batch, layer, d_model] capture, got shape {tuple(activations.shape)}; "
            "capture with position=last or position=mean, not position=all"
        )
    return activations.float()

def _layer_labels(layers: Optional[Sequence[int]], count: int) -> List[int]:
    """Use the caller's layer indices, or fall back to positions within the capture"""
    if layers is None:
        return list(range(count))
    layers = list(layers)
    if len(layers) != count:
        raise VizError(f"capture has {count} layers but {len(layers)} layer indices were given")
    return layers

def plot_layer_norms(activations: torch.Tensor, layers: Optional[Sequence[int]] = None, ax=None):
    """Mean residual stream norm at each captured layer, with the spread across prompts

    The cheapest check there is. Norms should climb with depth; a layer sitting
    at zero captured nothing, and a flat line means every layer captured the
    same thing.
    """
    set_style()
    activations = _as_batch_layer_model(activations)
    indices = _layer_labels(layers, activations.shape[1])
    norms = activations.norm(dim=-1)
    means = norms.mean(dim=0).tolist()
    lower = norms.quantile(0.1, dim=0).tolist()
    upper = norms.quantile(0.9, dim=0).tolist()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(indices, means, marker="o", color=PALETTE["primary"], linewidth=2)
    ax.fill_between(indices, lower, upper, color=PALETTE["primary"], alpha=0.18, label="10th-90th percentile")
    ax.legend()
    ax.set_xlabel("Layer")
    ax.set_ylabel("Residual stream norm")
    ax.set_title(f"Activation norm by layer, over {activations.shape[0]} prompts")
    return ax

def plot_activation_heatmap(
    activations: torch.Tensor,
    layers: Optional[Sequence[int]] = None,
    example: int = 0,
    dims: int = 128,
    ax=None,
):
    """Heatmap of one prompt's residual stream, layers down, dimensions across

    Only the first `dims` dimensions are drawn: a residual stream is hundreds
    of dimensions wide at the small end and thousands at the large one, and a
    strip that wide is a smear rather than a chart. The dimensions that survive are a fixed prefix rather than a random
    sample, so two captures of the same model are comparable side by side.

    What to look for: a vertical stripe is a dimension that fires at every
    depth -- residual streams have a handful of these outlier dimensions, and
    they are the ones that dominate an unnormalized distance.
    """
    set_style()
    activations = _as_batch_layer_model(activations)
    if not 0 <= example < activations.shape[0]:
        raise VizError(f"example {example} is outside the {activations.shape[0]} prompts captured")
    indices = _layer_labels(layers, activations.shape[1])
    width = min(dims, activations.shape[2])
    matrix = activations[example, :, :width]

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 0.32 * len(indices) + 2))

    limit = float(matrix.abs().quantile(0.99))
    sns.heatmap(matrix.numpy(), cmap=DIVERGING, center=0, vmin=-limit, vmax=limit, ax=ax,
                yticklabels=indices, xticklabels=False, cbar_kws={"label": "activation"})
    ax.set_xlabel(f"Residual stream dimension (first {width} of {activations.shape[2]})")
    ax.set_ylabel("Layer")
    ax.set_title(f"Residual stream of prompt {example}, clipped at the 99th percentile")
    return ax

def plot_token_layer_norms(
    activations: torch.Tensor,
    tokens: Sequence[str],
    layers: Optional[Sequence[int]] = None,
    example: int = 0,
    normalize: str = "layer",
    ax=None,
):
    """Heatmap of norm at every (layer, token position) for one prompt

    Needs a capture taken with position=all, which is the only way to see
    where in the sequence the model is doing work rather than only what it
    ended up with at the last token.

    normalize="layer" scales each row to its own maximum, and is the default
    for a reason worth knowing about: transformers park an enormous norm on
    the first token and use it as an attention sink, often an order of
    magnitude above every other position. Drawn raw, that one cell owns the
    colour scale and the rest of the prompt is a black rectangle. The sink is
    real and the row maximum is printed in the title so it is not hidden --
    but it is one fact, and it should not cost you the other seven.

    Pass normalize="none" to see the absolute magnitudes, sink and all.
    """
    set_style()
    if activations.dim() != 4:
        raise VizError(
            f"expected a [batch, layer, seq, d_model] capture, got shape {tuple(activations.shape)}; "
            "capture with position=all"
        )
    if normalize not in ("layer", "none"):
        raise VizError(f"normalize must be 'layer' or 'none', got '{normalize}'")
    indices = _layer_labels(layers, activations.shape[1])
    matrix = activations[example].float().norm(dim=-1)
    tokens = list(tokens)[: matrix.shape[1]]
    matrix = matrix[:, : len(tokens)]
    peak = float(matrix.max())

    if normalize == "layer":
        matrix = matrix / matrix.max(dim=1, keepdim=True).values.clamp_min(1e-12)
        legend = "norm, relative to the layer's own maximum"
    else:
        legend = "norm"

    if ax is None:
        _, ax = plt.subplots(figsize=(0.55 * len(tokens) + 3, 0.32 * len(indices) + 2))

    sns.heatmap(matrix.numpy(), cmap=SEQUENTIAL, ax=ax, yticklabels=indices,
                xticklabels=tokens, cbar_kws={"label": legend})
    ax.set_xlabel("Token")
    ax.set_ylabel("Layer")
    ax.tick_params(axis="x", rotation=60)
    ax.set_title(f"Norm by position and depth, prompt {example} (peak {peak:.0f})")
    return ax

def pca(features: torch.Tensor, components: int = 2) -> Tuple[torch.Tensor, List[float]]:
    """Project features onto their top principal components, with variance explained

    Computed by SVD on the centred matrix rather than by eigendecomposing a
    covariance, which is the numerically stable way round when d_model is far
    larger than the number of examples -- which it always is here.
    """
    features = features.float()
    centred = features - features.mean(dim=0, keepdim=True)
    _, singular, right = torch.linalg.svd(centred, full_matrices=False)
    take = min(components, right.shape[0])
    variance = singular.pow(2)
    ratios = (variance[:take] / variance.sum().clamp_min(1e-12)).tolist()
    return centred @ right[:take].T, ratios

def plot_layer_pca(
    activations: torch.Tensor,
    labels: Sequence[int],
    layers: Optional[Sequence[int]] = None,
    label_names: Optional[Sequence[str]] = None,
    columns: int = 4,
):
    """One PCA scatter per captured layer, coloured by class

    This is the probe sweep before the probe: the layer where the two colours
    come apart is the layer the sweep will pick. Two dimensions is less than a
    probe gets to use, so separation here is sufficient evidence of a linear
    signal and its absence is not proof there is none.

    Returns a Figure, since there is one panel per layer.
    """
    set_style()
    activations = _as_batch_layer_model(activations)
    indices = _layer_labels(layers, activations.shape[1])
    if len(labels) != activations.shape[0]:
        raise VizError(f"{activations.shape[0]} activations but {len(labels)} labels")

    names = tuple(label_names or ("negative", "positive"))
    colors = class_colors(names)
    rows = math.ceil(len(indices) / columns)
    figure, axes = plt.subplots(rows, min(columns, len(indices)),
                                figsize=(3.2 * min(columns, len(indices)), 3.1 * rows),
                                squeeze=False)

    label_tensor = torch.as_tensor(list(labels))
    for position, layer in enumerate(indices):
        ax = axes[position // columns][position % columns]
        coordinates, ratios = pca(activations[:, position])
        for value, name in enumerate(names):
            mask = label_tensor == value
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=18, alpha=0.75,
                       color=colors[name], label=name, edgecolors="none")
        explained = sum(ratios)
        ax.set_title(f"layer {layer}  ({explained:.0%} var)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for position in range(len(indices), rows * columns):
        axes[position // columns][position % columns].axis("off")

    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=len(names), frameon=False)
    figure.suptitle("Class separation in the top two principal components, by layer")
    figure.tight_layout(rect=(0, 0.05, 1, 0.97))
    return figure

def plot_layer_similarity(activations: torch.Tensor, layers: Optional[Sequence[int]] = None, ax=None):
    """Cosine similarity between every pair of captured layers, averaged over prompts

    The residual stream is additive, so neighbouring layers stay nearly
    parallel and the matrix is a bright diagonal band. How wide that band is
    says how fast the representation turns over; a sharp edge is a depth where
    the model changes what it is computing.
    """
    set_style()
    activations = _as_batch_layer_model(activations)
    indices = _layer_labels(layers, activations.shape[1])
    normalized = activations / activations.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # [batch, layer, layer] pairwise cosines, then averaged over the batch
    similarity = torch.einsum("bld,bmd->blm", normalized, normalized).mean(dim=0)

    if ax is None:
        _, ax = plt.subplots(figsize=(0.42 * len(indices) + 3, 0.38 * len(indices) + 2.5))

    sns.heatmap(similarity.numpy(), cmap=DIVERGING, center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=indices, yticklabels=indices, square=True,
                cbar_kws={"label": "cosine similarity"})
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title("How fast the residual stream turns over")
    return ax

def plot_drift(
    reference: torch.Tensor,
    other: torch.Tensor,
    layers: Optional[Sequence[int]] = None,
    reference_name: str = "reference",
    other_name: str = "compared",
    ax=None,
):
    """Per-layer agreement between two captures of the same prompts

    Cosine similarity on the left axis, relative L2 error on the right. Both
    are needed: quantization can leave a direction almost untouched while
    changing its magnitude, and a cosine of 0.999 next to a 15% norm error is
    a different finding from 0.999 next to 1%.
    """
    set_style()
    reference = _as_batch_layer_model(reference)
    other = _as_batch_layer_model(other)
    if reference.shape != other.shape:
        raise VizError(
            f"captures must have the same shape to be compared, got "
            f"{tuple(reference.shape)} and {tuple(other.shape)}"
        )
    indices = _layer_labels(layers, reference.shape[1])

    cosine = torch.nn.functional.cosine_similarity(reference, other, dim=-1).mean(dim=0).tolist()
    relative = ((other - reference).norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-12)).mean(dim=0).tolist()

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 4.8))

    ax.plot(indices, cosine, marker="o", color=PALETTE["primary"], linewidth=2, label="cosine similarity")
    ax.set_ylabel("Cosine similarity", color=PALETTE["primary"])
    ax.set_xlabel("Layer")
    ax.tick_params(axis="y", labelcolor=PALETTE["primary"])

    twin = ax.twinx()
    twin.plot(indices, relative, marker="s", color=PALETTE["steered"], linewidth=2,
              linestyle="--", label="relative L2 error")
    twin.set_ylabel("Relative L2 error", color=PALETTE["steered"])
    twin.tick_params(axis="y", labelcolor=PALETTE["steered"])
    twin.grid(False)

    ax.set_title(f"Activation drift: {other_name} against {reference_name}")
    return ax
