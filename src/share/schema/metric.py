from dataclasses import dataclass
from typing import Any, Dict

"""
A number that cannot be separated from the definition that produced it.

`faithfulness` names at least two different quantities in this field: the
logit-difference recovery under restoration that this repository measures, and
the normalized KL reproduction the faithfulness critique asks for. Both report
as a fraction near 0.9, neither is wrong, and an artifact that says only 0.919
can be compared with neither.

Metric is to a number what Payload is to a tensor, for exactly the same
reason -- and the earlier version of this format enforced it for every tensor
and then let the headline claim travel as a bare float.

A common pipe could be: verify | Metric | show
"""

@dataclass(frozen=True)
class Metric:
    """A number that cannot be separated from the definition that produced it

    `definition` is prose and `units` is prose, for the same reason
    Payload.units is: an enum is a list every new measurement has to be added
    to before it can be shared.

    `definition` has no option list by construction -- it is a sentence saying
    what was computed, and a definition that could be picked from a menu would
    not have needed writing. What it must do is distinguish this quantity from
    the other one with the same name: "recovery of the clean logit difference
    when only these heads are restored into the corrupted run" rather than
    "how faithful the circuit is". Empty is refused.

    `units` follows Payload's list, plus the ones only a scalar takes:

        "recovery"               a fraction of the span between the baselines
        "logits"                 a logit difference, unnormalized
        "heads"                  a count of components
        "auc"                    area under the ROC curve, in [0, 1]
        "share"                  a proportion, in [0, 1]
        "examples"               a count of rows measured over
        "loss"                   a training objective, not a held-out number
        "activation"             a norm in residual-stream coordinates
        "mean activation norms"  a steering strength, scaled to the layer
        "unspecified"            the default, and a thing to fix
    """
    value: float
    definition: str
    units: str = "unspecified"

    def describe(self) -> Dict[str, Any]:
        """The card's entry for this metric: what it is, and what it is in"""
        return {"value": float(self.value), "definition": self.definition, "units": self.units}
