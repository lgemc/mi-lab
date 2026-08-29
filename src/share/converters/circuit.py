from typing import Dict, Optional, Sequence

import torch

from ...core.config import ModelConfig
from ...data.ioi import IOIDataset
from ...methods.circuits import Attribution, CircuitReport, HeadEffects, HeadRoles, PatchGrid
from ..schema import Artifact, Controls, Metric, Node, Payload, Site, Span
from .common import model_ref

"""
A finished circuit study, packaged as the graph plus both halves' grids.

The nodes are the circuit and nothing else. The full grids stay as tensors, so
a reader who disagrees with the threshold can redo the selection from the same
numbers instead of taking this one on trust -- which is the difference between
sharing a result and sharing a claim about one.

A common pipe could be: discover | verify | from_circuit | save
"""

LOGIT_DIFFERENCE = "logit_difference"

# What each number in the report actually is. `faithfulness` is the one that
# has to be spelled out: it names a logit-difference recovery under
# restoration here and a normalized KL reproduction in the faithfulness
# literature, both land near 0.9, and they are not the same quantity.
DEFINITIONS = {
    "faithfulness": (
        "recovery of the clean logit difference when only these heads are restored into the "
        "corrupted run; 1.0 is the clean baseline, 0.0 the corrupted one"
    ),
    "necessity": (
        "1 - the recovery left when these heads alone are corrupted in an otherwise clean run; "
        "1.0 means the rest of the model does not do this without them"
    ),
    "n_heads": "how many heads the greedy search kept",
    "threshold": "the smallest gain in cumulative recovery for which the search adds another head",
    "attribution_remainder": (
        "logit difference left over after summing every component's direct write through the frozen "
        "unembedding; a receipt on the decomposition, not a result"
    ),
}

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
        model=model_ref(cfg, cfg.id),
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
            "faithfulness": Metric(report.faithfulness, DEFINITIONS["faithfulness"], "recovery"),
            "necessity": Metric(report.necessity, DEFINITIONS["necessity"], "recovery"),
            "n_heads": Metric(float(len(report.circuit)), DEFINITIONS["n_heads"], "heads"),
            "threshold": Metric(report.circuit.threshold, DEFINITIONS["threshold"], "recovery"),
            "attribution_remainder": Metric(
                attribution.residual, DEFINITIONS["attribution_remainder"], "logits"
            ),
        },
        span=span,
        # nothing here ablates this circuit against another task, so both slots ship
        # empty rather than absent: a circuit measured only on its own task is not
        # shown to be about that task, and the reader has to be able to see that
        controls=Controls(),
        nodes=nodes,
        edges=[],
        tensors=tensors,
        notes=(
            "Attribution is the direct path only and patching is causal; where the two disagree the "
            "disagreement is the result, so both are stored per head rather than one summary score. "
            "No edges were measured. No cross-task ablation was run either: every number here is "
            "within-task, so this says these heads carry the task and not that they are particular "
            "to it -- circuits at this level are largely shared infrastructure."
        ),
    ).validate()
