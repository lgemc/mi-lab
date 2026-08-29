from dataclasses import replace
from typing import Optional

import torch

from ...core.config import ModelConfig
from ...methods.probing import LinearProbe
from ..schema.artifact import Artifact
from ..schema.metric import Metric
from ..schema.payload import Payload
from ..schema.site import Site
from .common import model_ref

"""
A probe, packaged as something another lab can actually apply, and read back.

The standardization travels with the direction: a probe shipped as weights
alone is a vector in coordinates the receiver does not have, and applying it
looks like it works. to_probe sits here beside from_probe because the round
trip is the test of the whole format -- a shared artifact that does not come
back as a working object has shared nothing.

A common pipe could be: train_probe | from_probe | save | load | to_probe
"""

# What a probe's numbers are, so a shared one is comparable with a shared one
# from somewhere else. Anything a caller added that is not here is packaged
# with its method named as the definition, which is weak but true -- and
# visible, which is the point of the field.
DEFINITIONS = {
    "auc": ("area under the ROC curve of the probe's score against the held-out labels", "auc"),
    "accuracy": ("share of held-out examples the probe scores on the correct side of zero", "share"),
    "n": ("held-out examples the other metrics were measured over", "examples"),
    "train_loss": ("final training objective on the fitting set, not a held-out number", "loss"),
}

def _metric(name: str, value: float, method: str) -> Metric:
    """Package one of a probe's numbers with what it is, refusing to ship a bare float"""
    definition, units = DEFINITIONS.get(
        name, (f"unstated by the tool that measured it; recorded by '{method}' as '{name}'", "unspecified")
    )
    return Metric(value=float(value), definition=definition, units=units)

def from_probe(probe: LinearProbe, cfg: Optional[ModelConfig] = None, name: Optional[str] = None) -> Artifact:
    """Package a probe as something another lab can actually apply

    The standardization travels with the direction. A probe shipped as weights
    alone is a vector in coordinates the receiver does not have, and applying
    it looks like it works.
    """
    model = model_ref(cfg, probe.model_id)
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
        metrics={name: _metric(name, value, probe.method) for name, value in probe.metrics.items()},
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
        metrics=dict(artifact.metric_values),
    )
