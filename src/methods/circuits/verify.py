"""What a circuit is worth, asked four ways, because no one of them is the question.

A set of heads that reproduces the behaviour is not yet a claim about the
model. Four checks stand between the two, and each one catches something the
others pass:

    faithfulness   restore only these heads into the corrupted run. Are they
                   enough on their own?
    necessity      write these heads back to their corrupted values in a clean
                   run. Does the behaviour collapse without them, or is this
                   one of several routes?
    minimality     per head, how much faithfulness drops when it alone goes.
                   A head near zero is a passenger greed picked up.
    completeness   take a subset K out of the circuit and the *same* K out of
                   the model, and see whether they break together.

The fourth is the one the first three cannot express, and it is the one Wang
et al. named. A circuit can be faithful and incomplete -- reproducing the
behaviour by a route the model does not use -- and the gap is exactly where a
component the circuit left out is doing work. It is sampled rather than
exhaustive because 2^n subsets is not a thing to run, and the drawn subsets
are kept so the number ships with the evidence behind it.

`verify`'s minimality is weaker than completeness in the direction that
matters: it drops one head at a time, so it misses a pair that is redundant
together and load-bearing apart. That is why both are here rather than one.

A common pipe could be: discover | verify | spare | completeness
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from ...data.tasks import CircuitTask
from ...model.adapter import require_circuits
from ..common.components import HeadId
from ..common.errors import CircuitError
from ..common.intervention import head_patch
from ..common.span import Baselines, baselines, mean_logit_difference
from .patching import restore
from .search import Circuit


@dataclass
class CircuitReport:
    """What a circuit is worth: is it enough, is it needed, is any of it spare"""
    circuit: Circuit
    faithfulness: float
    necessity: float
    minimality: Dict[HeadId, float]
    baselines: Baselines

    def spare(self, tolerance: float = 0.05) -> List[HeadId]:
        """Heads whose removal costs the circuit almost nothing"""
        return [head for head, drop in self.minimality.items() if drop < tolerance]

    def __str__(self) -> str:
        return (
            f"{len(self.circuit)} heads  faithfulness {self.faithfulness:.2f}  "
            f"necessity {self.necessity:.2f}  spare {self.spare()}"
        )


def verify(adapter, dataset: CircuitTask, circuit: Circuit) -> CircuitReport:
    """Check a circuit three ways: sufficient, needed, and free of passengers

    - faithfulness: restore only these heads into the corrupted run. 1.0 means
      they are enough on their own.
    - necessity: do the opposite -- take a clean run and write *these* heads
      back to their corrupted values. 1.0 means the clean behaviour collapses
      without them, so they are not merely one of several routes to the answer.
    - minimality: per head, how much faithfulness drops when it alone is
      dropped. A head near zero is a passenger the greedy search picked up.

    This is weaker than the completeness test in Wang et al., which checks
    every subset rather than every single head, and it is weaker in the
    direction that matters: it can miss a pair of heads that are redundant
    together and load-bearing apart.
    """
    adapter = require_circuits(adapter)
    if not circuit.heads:
        raise CircuitError("an empty circuit has nothing to verify")
    reference = baselines(adapter, dataset)
    clean_donors = adapter.head_outputs(dataset.clean)
    corrupted_donors = adapter.head_outputs(dataset.corrupted)

    faithfulness = restore(adapter, dataset, reference, circuit.heads, clean_donors)

    with adapter.patch(heads=head_patch(circuit.heads, corrupted_donors)):
        broken = mean_logit_difference(adapter, dataset.clean, reference.io, reference.subject)

    minimality = {}
    for head_id in circuit.heads:
        rest = [other for other in circuit.heads if other != head_id]
        without = restore(adapter, dataset, reference, rest, clean_donors)
        minimality[head_id] = faithfulness - without

    return CircuitReport(
        circuit=circuit,
        faithfulness=faithfulness,
        necessity=1.0 - reference.recovery(broken),
        minimality=minimality,
        baselines=reference,
    )


@dataclass
class Completeness:
    """How closely the circuit stands in for the model when the same parts leave both

    Faithfulness asks whether the circuit is enough and minimality asks
    whether any one head is spare. Completeness asks the harder question
    between them: take some subset K out of the circuit, take the *same* K out
    of the whole model, and see whether the two break together. A circuit that
    is faithful and incomplete is one that reproduces the behaviour by a route
    the model does not use -- the gap is where a component the circuit left
    out is doing work.

    Sampled rather than exhaustive: 2^n subsets is not a thing to run, and the
    subsets that were drawn are kept so the number can be quoted with the
    evidence behind it. `incompleteness` is the worst gap found, which is the
    direction the claim has to survive.
    """
    subsets: List[List[HeadId]]
    circuit_scores: List[float]
    model_scores: List[float]

    @property
    def gaps(self) -> List[float]:
        """How far the circuit and the model drifted apart on each subset"""
        return [abs(left - right) for left, right in zip(self.circuit_scores, self.model_scores, strict=True)]

    @property
    def incompleteness(self) -> float:
        """The worst gap any sampled subset opened, which is what the claim has to survive"""
        return max(self.gaps) if self.gaps else 0.0

    @property
    def mean_gap(self) -> float:
        return sum(self.gaps) / len(self.gaps) if self.gaps else 0.0

    def __str__(self) -> str:
        return f"{len(self.subsets)} subsets  worst gap {self.incompleteness:.3f}  mean {self.mean_gap:.3f}"


def completeness(
    adapter, dataset: CircuitTask, circuit: Circuit, samples: int = 8, seed: int = 0,
    reference: Optional[Baselines] = None,
) -> Completeness:
    """Sample subsets of the circuit and check that removing one hurts circuit and model alike

    For each drawn subset K: restore the circuit minus K into the corrupted
    run, and separately knock K out of the clean run. Both land on the same
    recovery scale and both sit near 1.0 when K did not matter, so the gap
    between them is the quantity of interest and nothing has to be rescaled to
    compare them.

    The empty subset is always drawn first, because that pair is the circuit's
    own faithfulness against an untouched model and it is the anchor the rest
    of the curve is read against.
    """
    adapter = require_circuits(adapter)
    if not circuit.heads:
        raise CircuitError("an empty circuit has no subsets to check for completeness")
    if samples < 1:
        raise CircuitError(f"completeness needs at least one subset to sample, got samples={samples}")
    measured = reference or baselines(adapter, dataset)
    clean_donors = adapter.head_outputs(dataset.clean)
    corrupted_donors = adapter.head_outputs(dataset.corrupted)
    rng = random.Random(seed)

    subsets: List[List[HeadId]] = [[]]
    while len(subsets) < samples:
        size = rng.randint(1, len(circuit.heads))
        drawn = sorted(rng.sample(circuit.heads, size))
        if drawn not in subsets:
            subsets.append(drawn)
        elif len(subsets) >= 2 ** len(circuit.heads):
            break

    circuit_scores, model_scores = [], []
    for removed in subsets:
        kept = [head for head in circuit.heads if head not in removed]
        circuit_scores.append(restore(adapter, dataset, measured, kept, clean_donors))
        if not removed:
            model_scores.append(1.0)
            continue
        with adapter.patch(heads=head_patch(removed, corrupted_donors)):
            damaged = mean_logit_difference(adapter, dataset.clean, measured.io, measured.subject)
        model_scores.append(measured.recovery(damaged))
    return Completeness(subsets=subsets, circuit_scores=circuit_scores, model_scores=model_scores)
