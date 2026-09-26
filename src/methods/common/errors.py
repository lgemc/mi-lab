"""Every way this layer refuses, as one tree, because the tree was the thing nobody could see.

Each subsystem here raises its own `ValueError` subclass, and until this module
each one defined that subclass next to the code that raised it. The base was
`CircuitError`, which lived in `circuits.py` -- so `cost`, `knockout`,
`neurons`, `components`, `gates` and `sheaves` all imported the circuit study
to get a class with no behaviour in it. Six modules depended on the largest
module in the package for a name, and reading the hierarchy meant opening six
files and hoping none of them had quietly picked a different parent.

So the tree is written down once, here, at the bottom of the layer where
nothing else lives. `MethodError` is what every refusal in `methods` is, and
it is what a caller catches to mean "the measurement was asked for something
it cannot mean". Under it:

    CircuitError    a claim about which parts of a model do a task. Everything
                    that names components inherits from it -- the discovery
                    techniques, the component vocabulary, the cost model, the
                    knockout, the gates and the sheaf -- because they are all
                    the same claim measured differently, and code that catches
                    one kind of wrong circuit usually wants all of them.
    ProbeError      a direction read off a residual stream. Not a circuit: a
                    probe names no components, so a caller filtering for
                    circuit failures must not catch it by accident.

`QualityError` is the one that hangs off somewhere else. It is a `MetricError`
from `core.metrics`, because a translation that cannot be scored against its
references is a failed measurement rather than a failed circuit, and the code
that catches a bad BLEU is the code that catches a bad AUC.

Nothing here carries state or behaviour. A subclass exists exactly when a
caller would want to catch that kind of failure and not the others; adding one
that no `except` ever names is adding a word, not a distinction.
"""

from ...core.metrics import MetricError


class MethodError(ValueError):
    """Raised when a measurement in this layer is asked for something it cannot mean"""


class CircuitError(MethodError):
    """Raised when a circuit measurement is asked for something it cannot mean: no span, no heads, no positions"""


class ComponentError(CircuitError):
    """A component name or set that does not describe this model"""


class CostError(CircuitError):
    """A cost that cannot be computed for the model or the set it was asked about"""


class DiscoveryError(CircuitError):
    """Raised when a technique is asked for something it cannot answer: no such method, no heads to select"""


class GateError(CircuitError):
    """A mask that does not describe this model, or a band that cannot be parsed"""


class KnockoutError(CircuitError):
    """A mean that does not belong to this model, or a component it cannot ablate"""


class NeuronError(CircuitError):
    """An MLP whose neurons cannot be found, or a contrast that does not line up"""


class SheafError(CircuitError):
    """Raised when gate training is asked for something it cannot train: no gateable weight, no holdout"""


class WiringError(CircuitError):
    """Raised when an edge set cannot be read as a circuit for the model in hand"""


class ProbeError(MethodError):
    """Raised when a probe is asked for something it cannot do: wrong width, one-class data, no examples"""


class QualityError(MetricError):
    """A comparison that cannot be made of the hypotheses it was given"""
