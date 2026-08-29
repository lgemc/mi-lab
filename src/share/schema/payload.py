from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

"""
A tensor that cannot be separated from what its axes mean.

`head_effects` is [layer, head] in recovery; `residual_patch` is
[layer, position]. A bare 2-D float array in a file named effects.pt is one the
next reader transposes, and the transposed heatmap still looks plausible. So
the values, the axis names, the unit and the tick labels are one object, and
there is no call anywhere that stores a tensor without saying what it means.

`labels` is what makes a heatmap redrawable by a tool that never saw the
prompts -- the difference between shipping a figure and shipping the thing the
figure was made of.

A common pipe could be: capture | Payload | save | redraw
"""

@dataclass(frozen=True)
class Payload:
    """A tensor that cannot be separated from what its axes mean

    `units` is prose on purpose, because the alternative is an enum that every
    new measurement has to be added to before it can be shared. It is not
    unconstrained in practice -- the ones this repository writes are:

        "recovery"                 a fraction of the span between the baselines
        "logits"                   a logit difference, unnormalized
        "activation"               raw residual-stream coordinates
        "standardized"             coordinates after (x - mean) / std
        "attention"                attention mass, in [0, 1]
        "probe score"              a probe's output on a steered continuation
        "mean activation norms"    a steering strength, scaled to the layer
        "non-repeated word share"  the fluency half of a steering sweep
        "unspecified"              the default, and a thing to fix

    A unit not on that list is fine and expected; a unit that is not written
    down at all is what this field exists to prevent.

    `axes` names the dimensions, and the names checked against the card are
    "layer" (against the site), "head" and "d_model" (against the model). Any
    other name -- "position", "point", "role" -- is carried but not checked,
    because only the card knows how long it should be.

    `labels` names the ticks along an axis: the token strings under a position
    axis, the role names under a role axis.
    """
    values: torch.Tensor
    axes: List[str] = field(default_factory=list)
    units: str = "unspecified"
    labels: Dict[str, List[str]] = field(default_factory=dict)

    def describe(self) -> Dict[str, Any]:
        """The manifest's entry for this tensor: its shape, dtype, axes and unit"""
        return {
            "shape": list(self.values.shape),
            "dtype": str(self.values.dtype).removeprefix("torch."),
            "axes": list(self.axes),
            "units": self.units,
            "labels": {axis: list(names) for axis, names in self.labels.items()},
        }
