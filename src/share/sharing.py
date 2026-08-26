from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..core.config import ModelConfig, load_config
from ..data.ioi import IOIDataset
from ..methods.circuits import Attribution, CircuitReport, HeadEffects, HeadRoles, PatchGrid
from ..methods.probing import LinearProbe
from .artifact import Artifact, ModelRef, Node, Payload, Site, Span

"""
Turning what this lab measures into artifacts anyone can load, and back.

core/artifact.py is the envelope and knows nothing about this repository;
this module is the one place that knows both, so a new experiment kind gets a
converter here rather than a special case inside the format. Keeping them
apart is what lets `Artifact.load` read a card without importing transformers.

Every converter is lossy in one direction on purpose. It writes down what
another lab needs to *use* the result -- the direction, the heads, the
baselines, the site -- and not the intermediate state that only means
something inside this process. What survives the round trip is what the format
claims to carry, which is why to_probe exists next to from_probe: a probe that
does not come back as a working probe is a file, not an artifact.

A common pipe could be: verify | from_circuit | save | load
"""

LOGIT_DIFFERENCE = "logit_difference"

def _model_ref(cfg: Optional[ModelConfig], model_id: str) -> ModelRef:
    """Describe the model an artifact was measured on, refusing to guess at it

    A ModelRef whose hf_name was invented is worse than no artifact: it names
    a checkpoint that will load and produce different numbers.
    """
    if cfg is not None:
        return ModelRef.from_config(cfg)
    try:
        return ModelRef.from_config(load_config(model_id))
    except ValueError as error:
        raise ValueError(
            f"'{model_id}' does not name a config, so the checkpoint behind this result cannot be written "
            "down; pass the ModelConfig it was measured on"
        ) from error

# --------------------------------------------------------------------- circuits

def from_circuit(
    cfg: ModelConfig,
    dataset: IOIDataset,
    attribution: Attribution,
    effects: HeadEffects,
    report: CircuitReport,
    roles: Optional[HeadRoles] = None,
    grid: Optional[PatchGrid] = None,
    tokens: Optional[Sequence[str]] = None,
    landmarks: Optional[Dict[str, int]] = None,
    name: Optional[str] = None,
) -> Artifact:
    """Package a finished circuit study: the graph, both halves' grids, and the span

    The nodes are the circuit and nothing else -- the heads the search kept,
    each carrying what attribution said about it, what patching said about it,
    and what dropping it costs. The full grids stay as tensors, so a reader
    who disagrees with the threshold can redo the selection from the same
    numbers instead of taking this one on trust.

    edges is empty and says so. This repository measures which heads matter,
    not which head feeds which, and an artifact that left the field out would
    read as a circuit whose connections nobody thought to record.
    """
    span = Span(metric=LOGIT_DIFFERENCE, clean=report.baselines.clean, corrupted=report.baselines.corrupted)
    named = roles.assign() if roles is not None else {}
    # the prompt's token strings describe the data whether or not a position map was
    # measured over them, so they are taken from wherever they are available
    positions = list(tokens if tokens is not None else (grid.tokens if grid is not None else []))
    marks = dict(landmarks if landmarks is not None else (grid.landmarks if grid is not None else {}))

    nodes = []
    for step, ((layer, head), cumulative) in enumerate(
        zip(report.circuit.heads, report.circuit.scores, strict=True), start=1
    ):
        row = effects.layers.index(layer)
        nodes.append(Node(
            id=f"L{layer}H{head}",
            component="head",
            layer=layer,
            head=head,
            role=named.get((layer, head)),
            in_circuit=True,
            scores={
                "attribution": float(attribution.heads[layer, head]),
                "causal": float(effects.effects[row, head]),
                "minimality": float(report.minimality[(layer, head)]),
                "cumulative_recovery": float(cumulative),
                "step": float(step),
            },
        ))

    # attribution answers for every layer in one pass while patching may have swept a
    # subset, so both grids are cut down to the layers the site actually names -- a row
    # index that means layer 4 in one tensor and layer 0 in the next is the bug the
    # site exists to prevent
    rows = torch.tensor(effects.layers, dtype=torch.long)
    tensors = {
        "head_attribution": Payload(
            values=attribution.heads.index_select(0, rows).float(), axes=["layer", "head"], units="logits"
        ),
        "head_effects": Payload(
            values=effects.effects.float(), axes=["layer", "head"], units="recovery"
        ),
        "mlp_attribution": Payload(
            values=attribution.mlps.index_select(0, rows).float(), axes=["layer"], units="logits"
        ),
    }
    if roles is not None:
        tensors["role_weights"] = Payload(
            values=roles.weights.index_select(0, rows).float(), axes=["layer", "head", "role"], units="attention",
            labels={"role": list(roles.roles)},
        )
    if grid is not None:
        if grid.layers != effects.layers:
            raise ValueError(
                f"the position map swept layers {grid.layers} and the head sweep swept {effects.layers}; "
                "one artifact names one site, so measure both over the same layers or package them separately"
            )
        tensors["residual_patch"] = Payload(
            values=grid.effects.float(), axes=["layer", "position"], units="recovery",
            labels={"position": positions},
        )

    return Artifact(
        kind="circuit",
        id=name or f"{dataset.name}-{cfg.id}",
        model=_model_ref(cfg, cfg.id),
        site=Site.at(effects.layers, cfg.n_layers or 0, component="head_out", position="all"),
        task={
            "name": dataset.name,
            "task": "indirect object identification",
            "frame": dataset.frame,
            "corruption": dataset.corruption,
            "n": len(dataset),
            "balance": dataset.balance,
            "tokens": positions,
            "landmarks": marks,
            "example": {
                "clean": dataset.examples[0].clean,
                "corrupted": dataset.examples[0].corrupted,
                "answer": dataset.examples[0].io,
                "distractor": dataset.examples[0].subject,
            } if len(dataset) else {},
        },
        method="direct_logit_attribution + activation_patching, greedy search",
        metrics={
            "faithfulness": report.faithfulness,
            "necessity": report.necessity,
            "n_heads": float(len(report.circuit)),
            "threshold": report.circuit.threshold,
            "attribution_remainder": attribution.residual,
        },
        span=span,
        nodes=nodes,
        edges=[],
        tensors=tensors,
        notes=(
            "Attribution is the direct path only and patching is causal; where the two disagree the "
            "disagreement is the result, so both are stored per head rather than one summary score. "
            "No edges were measured."
        ),
    ).validate()

# ----------------------------------------------------------------------- probes

def from_probe(probe: LinearProbe, cfg: Optional[ModelConfig] = None, name: Optional[str] = None) -> Artifact:
    """Package a probe as something another lab can actually apply

    The standardization travels with the direction. A probe shipped as weights
    alone is a vector in coordinates the receiver does not have, and applying
    it looks like it works.
    """
    model = _model_ref(cfg, probe.model_id)
    # the config is authoritative when it has been resolved against a checkpoint; the
    # probe's own record of the depth is the fallback, and it is what makes packing a
    # saved probe possible without loading the model it came from
    depth = model.n_layers or probe.n_layers
    if not depth:
        raise ValueError(
            f"probe '{probe.dataset}' records layer {probe.layer} but not how many layers '{probe.model_id}' "
            "has, so the depth it was read from cannot be written down; pass the resolved ModelConfig"
        )
    model = replace(model, n_layers=depth)
    return Artifact(
        kind="probe",
        id=name or f"{probe.dataset}-{probe.model_id}-L{probe.layer}",
        model=model,
        site=Site.at([probe.layer], depth, position=probe.position),
        task={"name": probe.dataset},
        method=probe.method,
        metrics=dict(probe.metrics),
        tensors={
            "weight": Payload(values=probe.weight.float(), axes=["d_model"], units="standardized"),
            "bias": Payload(values=torch.tensor(float(probe.bias)), axes=[], units="logits"),
            "mean": Payload(values=probe.mean.float(), axes=["d_model"], units="activation"),
            "std": Payload(values=probe.std.float(), axes=["d_model"], units="activation"),
            "direction": Payload(values=probe.direction.float(), axes=["d_model"], units="activation"),
        },
        notes=(
            "score(x) = ((x - mean) / std) @ weight + bias. Steer with 'direction' (weight / std), which is "
            "the same direction expressed in activation coordinates; steering with 'weight' reweights every "
            "dimension by this layer's activation scale."
        ),
    ).validate()

def to_probe(artifact: Artifact) -> LinearProbe:
    """Read a probe artifact back as a working probe

    The round trip is the test of the whole format: if a shared artifact does
    not come back as something that scores activations, nothing was shared.
    """
    if artifact.kind != "probe":
        raise ValueError(f"artifact '{artifact.id}' is a {artifact.kind}, not a probe")
    if not artifact.site.layers:
        raise ValueError(f"probe artifact '{artifact.id}' names no layer, so there is nowhere to apply it")
    return LinearProbe(
        weight=artifact.tensor("weight").double(),
        bias=float(artifact.tensor("bias")),
        mean=artifact.tensor("mean").double(),
        std=artifact.tensor("std").double(),
        layer=artifact.site.layers[0],
        model_id=artifact.model.id,
        n_layers=artifact.model.n_layers,
        position=artifact.site.position,
        dataset=str(artifact.task.get("name", "unnamed")),
        method=artifact.method,
        metrics=dict(artifact.metrics),
    )

def open_probe(path: str) -> LinearProbe:
    """Read a probe from either the .pt this repo writes or a shared .mia artifact

    Anything that only wants to *apply* a probe should not have to know which
    of the two it was handed. That is the whole promise of a shared format,
    and the cheapest place to keep it is here rather than in every caller.
    """
    source = Path(path)
    if source.is_dir():
        return to_probe(Artifact.load(str(source)))
    return LinearProbe.load(path)

# ------------------------------------------------------------- steering vectors

def from_steering(
    cfg: ModelConfig,
    vector: torch.Tensor,
    layer: int,
    source: str,
    points: Optional[Sequence[Any]] = None,
    name: Optional[str] = None,
    dataset: str = "unnamed",
) -> Artifact:
    """Package a steering direction together with the sweep that found its ceiling

    A direction on its own is untestable: it moves the model at every strength
    and the question was always which strengths keep the text intact. When a
    strength_sweep is handed over, its three columns ship with the vector, so
    the receiver reads the ceiling off the artifact instead of rediscovering it.
    """
    tensors = {
        "vector": Payload(values=vector.float(), axes=["d_model"], units="activation"),
    }
    metrics: Dict[str, float] = {"norm": float(vector.float().norm())}
    if points:
        tensors["strengths"] = Payload(
            values=torch.tensor([point.strength for point in points]), axes=["point"], units="mean activation norms"
        )
        tensors["effect"] = Payload(
            values=torch.tensor([point.effect for point in points]), axes=["point"], units="probe score"
        )
        tensors["fluency"] = Payload(
            values=torch.tensor([point.fluency for point in points]), axes=["point"], units="non-repeated word share"
        )
        metrics["max_strength"] = float(max(point.strength for point in points))

    return Artifact(
        kind="steering_vector",
        id=name or f"{dataset}-{cfg.id}-L{layer}",
        model=_model_ref(cfg, cfg.id),
        site=Site.at([layer], cfg.n_layers or 0, position="all"),
        task={"name": dataset, "source": source},
        method=source,
        metrics=metrics,
        tensors=tensors,
        notes=(
            "Added to the residual stream at this layer, scaled in mean activation norms of that layer's "
            "forward pass, so a strength means the same intervention size on another model. Any claim made "
            "with it needs a norm-matched random vector at the same layer as its control."
        ),
    ).validate()

# ------------------------------------------------------------- activation maps

def from_activations(
    cfg: ModelConfig,
    values: torch.Tensor,
    layers: Sequence[int],
    axes: Sequence[str],
    units: str = "activation",
    labels: Optional[Dict[str, List[str]]] = None,
    name: str = "activations",
    task: Optional[Dict[str, Any]] = None,
    position: str = "all",
) -> Artifact:
    """Package a map over layers and positions as data rather than as a picture

    Activation maps travel as figures today, which is why nobody can subtract
    two of them. The axes and their tick labels are required here for exactly
    that reason: a heatmap whose columns are named tokens can be redrawn,
    compared and differenced by a tool that never ran the model.
    """
    return Artifact(
        kind="activation_map",
        id=name,
        model=_model_ref(cfg, cfg.id),
        site=Site.at(layers, cfg.n_layers or 0, position=position),
        task=dict(task or {}),
        method="capture",
        tensors={"values": Payload(values=values.float(), axes=list(axes), units=units, labels=dict(labels or {}))},
    ).validate()
