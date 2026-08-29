from typing import Any, Dict, List, Optional, Sequence

import torch

from ...core.config import ModelConfig
from ..schema import Artifact, Payload, Site
from .common import model_ref

"""
A map over layers and positions, packaged as data rather than as a picture.

Activation maps travel as figures today, which is why nobody can subtract two
of them. The axes and their tick labels are required for exactly that reason:
a heatmap whose columns are named tokens can be redrawn, compared and
differenced by a tool that never ran the model.

A common pipe could be: capture | from_activations | save
"""

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
        model=model_ref(cfg, cfg.id),
        site=Site.at(layers, cfg.n_layers or 0, position=position),
        task=dict(task or {}),
        method="capture",
        tensors={"values": Payload(values=values.float(), axes=list(axes), units=units, labels=dict(labels or {}))},
    ).validate()
