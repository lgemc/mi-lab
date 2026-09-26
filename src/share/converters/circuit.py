from typing import Any, Dict, Optional, Sequence

import torch

from ...core.config import ModelConfig
from ...data.tasks import CircuitTask
from ...methods.circuits.attribution import Attribution
from ...methods.circuits.patching import HeadEffects, PatchGrid
from ...methods.circuits.verify import CircuitReport
from ..definitions import describe
from ..schema.artifact import Artifact
from ..schema.controls import Controls
from ..schema.metric import Metric
from ..schema.node import Node
from ..schema.payload import Payload
from ..schema.site import Site
from ..schema.span import Span
from ..schema.vocabulary import Component, NodeComponent, Position
from .common import model_ref

"""
A finished circuit study, packaged as the graph plus both halves' grids.

The nodes are the circuit and nothing else. The full grids stay as tensors, so
a reader who disagrees with the threshold can redo the selection from the same
numbers instead of taking this one on trust -- which is the difference between
sharing a result and sharing a claim about one.

A common pipe could be: discover | verify | from_circuit | save
"""

#: The span's `metric` is the name of the readout the study actually ran under,
#: read off the Baselines rather than written down here. A constant was right
#: while there was one readout and is a second opinion the moment there are two.

def from_circuit(
    cfg: ModelConfig,
    dataset: CircuitTask,
    attribution: Attribution,
    effects: HeadEffects,
    report: CircuitReport,
    roles=None,
    grid: Optional[PatchGrid] = None,
    tokens: Optional[Sequence[str]] = None,
    landmarks: Optional[Dict[str, int]] = None,
    name: Optional[str] = None,
    description: Optional[Dict[str, Any]] = None,
) -> Artifact:
    """Package a finished circuit study: the graph, both halves' grids, and the span

    The nodes are the circuit and nothing else -- the heads the search kept,
    each carrying what attribution said about it, what patching said about it,
    and what dropping it costs. The full grids stay as tensors, so a reader
    who disagrees with the threshold can redo the selection from the same
    numbers instead of taking this one on trust.

    `description` is what the caller wants written on the task card beyond the
    name and size every CircuitTask has -- IOI's frame, its corruption, the two
    names one prompt is between. It is a parameter rather than something read
    off the dataset because this module is the format's side of the fence and
    the fields worth quoting are the task's: an IOI dataset has an answer and a
    distractor, a patch-grid task would have neither, and a converter that
    reached for `.io` would be a kernel module that knows what a name is.

    `roles` is left unannotated for the same reason. It is whatever the domain's
    role classifier produced, and all that is asked of it is `assign()` mapping
    a head to a label, plus `weights` and `roles` for the grid.

    edges is empty and says so. This repository measures which heads matter,
    not which head feeds which, and an artifact that left the field out would
    read as a circuit whose connections nobody thought to record.
    """
    readout = report.baselines.readout
    span = Span(metric=readout.name, clean=report.baselines.clean, corrupted=report.baselines.corrupted)
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
            component=NodeComponent.HEAD,
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
            values=attribution.heads.index_select(0, rows).float(), axes=["layer", "head"],
            units=readout.units,
        ),
        "head_effects": Payload(
            values=effects.effects.float(), axes=["layer", "head"], units="recovery"
        ),
        "mlp_attribution": Payload(
            values=attribution.mlps.index_select(0, rows).float(), axes=["layer"], units=readout.units,
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
        site=Site.at(effects.layers, cfg.n_layers or 0, component=Component.HEAD_OUT, position=Position.ALL),
        task={
            "name": dataset.name,
            "modality": dataset.modality,
            "n": len(dataset),
            "tokens": positions,
            "landmarks": marks,
            **(description or {}),
        },
        method="direct_logit_attribution + activation_patching, greedy search",
        metrics={
            **{
                name: Metric(value, *describe(name, score=readout))
                for name, value in (
                    ("faithfulness", report.faithfulness),
                    ("necessity", report.necessity),
                    ("n_heads", float(len(report.circuit))),
                    ("threshold", report.circuit.threshold),
                )
            },
            # the receipt is in whatever the readout is in, so its units come off
            # the readout rather than off a table that was written when there was
            # only one of them
            "attribution_remainder": Metric(
                attribution.residual, describe("attribution_remainder")[0], readout.units
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
