from typing import Any, Dict, Optional, Sequence

import torch

from ...core.config import ModelConfig
from ..schema import Artifact, Payload, Site
from .common import model_ref

"""
A steering direction, packaged with the sweep that found its ceiling.

A direction on its own is untestable: it moves the model at every strength and
the question was always which strengths keep the text intact. When a
strength_sweep is handed over its three columns ship with the vector, so the
receiver reads the ceiling off the artifact instead of rediscovering it.

A common pipe could be: strength_sweep | from_steering | save
"""

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
        model=model_ref(cfg, cfg.id),
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
