"""Faithfulness as a surface over methodology, not a number.

Miller, Chughtai and Saunders (2407.08734) measured the same published circuits
under varied ablation methodology and found the scores moved more than the
circuits did: the IOI circuit reads ~87% under node-and-mean ablation and over
100% under edge-and-mean, resample sits systematically below mean, and the
per-example interquartile range spans roughly half the scale. Their conclusion
is that a faithfulness score reports the researcher's choices as much as the
components, and that the task a circuit is asked to perform is *defined* by the
ablation used to test it.

So this module refuses to return one number. `Methodology` is the six-tuple
that paper enumerates -- granularity, component type, ablation value, token
positions, direction, and which set gets ablated -- plus the aggregation order,
which is a seventh they show matters and which no paper before them stated.
Every score carries the tuple that produced it, and `sensitivity` walks the
axes so the spread is the output rather than a caveat.

Two aggregation orders, because they disagree and both are in the literature:

    ratio_of_means   mean(F) / mean(M)      Wang et al.
    mean_of_ratios   mean(F / M)

The second is undefined wherever an example's full-model logit difference is
near zero, and it is the one that reveals per-example variance. Both are
reported; neither is called the real one.

`span` is this repo's own definition -- recovery between the corrupted and
clean baselines rather than a bare ratio to the full model -- kept because
every other number in this codebase is in those units and a faithfulness that
could not be compared with them would be a third convention.

A common pipe could be: circuit | methodology | ablate | differences | aggregate

What this module does not do: edges, branches, neurons, or position-restricted
ablation. The repo scores whole heads as nodes over all positions, so four of
the paper's axes have exactly one setting here, and they are recorded as that
setting rather than omitted -- an artifact that says `component: node` and one
that says nothing are different claims.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..core.metrics import logit_difference
from ..data.tasks import CircuitTask
from ..model.adapter import require_circuits
from .circuits import Baselines, CircuitError, HeadId, baselines

ABLATION_VALUES = ("zero", "mean", "resample")
DIRECTIONS = ("restore", "destroy")
SETS = ("circuit", "complement")
AGGREGATIONS = ("ratio_of_means", "mean_of_ratios", "span")

@dataclass(frozen=True)
class Methodology:
    """Everything about a faithfulness measurement that is a choice rather than a finding

    Named after the six axes of 2407.08734 with aggregation added. Frozen and
    carried on every result, because the paper's central finding is that two
    scores computed under different tuples are not comparable, and the only
    defence is to make the tuple travel with the number.
    """
    value: str = "resample"
    direction: str = "restore"
    ablated: str = "circuit"
    aggregation: str = "ratio_of_means"
    granularity: str = "head"
    component: str = "node"
    positions: str = "all"

    def __post_init__(self):
        for field, allowed in (("value", ABLATION_VALUES), ("direction", DIRECTIONS),
                               ("ablated", SETS), ("aggregation", AGGREGATIONS)):
            if getattr(self, field) not in allowed:
                raise CircuitError(f"{field} must be one of {allowed}, got '{getattr(self, field)}'")

    def label(self) -> str:
        return f"{self.value}/{self.direction}/{self.ablated}/{self.aggregation}"

@dataclass(frozen=True)
class Faithfulness:
    """One score, the tuple that produced it, and how far it moves per example

    The distribution is not decoration. 2407.08734 reports an interquartile
    range near half the scale on IOI, which means a mean faithfulness of 0.87
    is compatible with a circuit that reproduces almost nothing on a quarter of
    its inputs. A single number hides exactly the failure the measurement
    exists to catch, so worst is reported beside mean and both are in the
    dataclass rather than available on request.
    """
    score: float
    methodology: Methodology
    per_example: List[float]
    median: float
    iqr: Tuple[float, float]
    worst: float
    best: float
    n: int

    def __str__(self) -> str:
        low, high = self.iqr
        return (f"{self.methodology.label():<42} {self.score:>7.3f}  "
                f"median {self.median:>6.3f}  IQR [{low:>6.3f}, {high:>6.3f}]  worst {self.worst:>7.3f}")

def _differences(adapter, prompts: Sequence[str], reference: Baselines) -> torch.Tensor:
    """Per-example logit difference, not the batch mean

    The mean is what every existing helper here returns and it cannot express
    either of the two aggregation orders, because mean(F)/mean(M) and
    mean(F/M) need the vector to tell apart.
    """
    return logit_difference(adapter.logits(list(prompts)), reference.io, reference.subject).float().cpu()

def _off_value(value: str, clean: torch.Tensor, corrupted: torch.Tensor,
               layer: int, head: int) -> torch.Tensor:
    """What a head's output becomes when it is ablated, under each of the paper's values

    zero is the arbitrary baseline -- it removes the component *and* whatever
    the model does with a vector it never sees. mean removes what varied across
    the batch and leaves the constant part, so it asks what the circuit does
    given the context rather than from nothing. resample writes the
    counterfactual run's own activation, which is the only one of the three
    that keeps the model on its own distribution.
    """
    if value == "zero":
        return torch.zeros_like(clean[:, layer, head])
    if value == "mean":
        # the batch mean, broadcast back over the batch: constant across examples,
        # which is what makes it remove variation rather than the component
        return clean[:, layer, head].mean(dim=0, keepdim=True).expand_as(clean[:, layer, head])
    return corrupted[:, layer, head]

def _all_heads(adapter) -> List[HeadId]:
    return [(layer, head) for layer in range(adapter.cfg.n_layers) for head in range(adapter.cfg.n_heads)]

def measure(adapter, task: CircuitTask, circuit_heads: Sequence[HeadId],
            methodology: Optional[Methodology] = None,
            reference: Optional[Baselines] = None) -> Faithfulness:
    """One faithfulness score under one explicit methodology

    direction and `ablated` together pick which of four experiments this is:

      restore/circuit      the circuit's clean activations into a corrupted run.
                           Classic sufficiency: 1.0 means these heads alone
                           carry the behaviour.
      destroy/circuit      the circuit's ablated values into a clean run.
                           Necessity: low means the model needs them.
      destroy/complement   ablate everything else in a clean run -- running the
                           circuit alone by deletion. The phase 1b extraction
                           arm, and the one that collapses when the complement
                           is most of the network.
      restore/complement   the complement's clean activations into a corrupted
                           run: what the behaviour reaches without the circuit.

    They are not four views of one quantity. restore/circuit and
    destroy/complement both claim to run "only the circuit" and disagree
    whenever the model's other components do anything the corrupted run does
    not already supply, which is most of the time.
    """
    adapter = require_circuits(adapter)
    methodology = methodology or Methodology()
    if not circuit_heads:
        raise CircuitError("an empty circuit has no faithfulness to measure")
    reference = reference or baselines(adapter, task)

    clean = adapter.head_outputs(task.clean)
    corrupted = adapter.head_outputs(task.corrupted)
    chosen = set(circuit_heads)
    target = list(chosen) if methodology.ablated == "circuit" else \
        [head for head in _all_heads(adapter) if head not in chosen]
    if not target:
        raise CircuitError(f"the {methodology.ablated} is empty, so there is nothing to write in")

    patch: Dict[int, Dict[int, torch.Tensor]] = {}
    for layer, head in target:
        if methodology.direction == "restore":
            patch.setdefault(layer, {})[head] = clean[:, layer, head]
        else:
            patch.setdefault(layer, {})[head] = _off_value(methodology.value, clean, corrupted, layer, head)
    source = task.corrupted if methodology.direction == "restore" else task.clean

    with adapter.patch(heads=patch):
        ablated = _differences(adapter, source, reference)
    full = _differences(adapter, task.clean, reference)

    ratios = (ablated / full).tolist()
    if methodology.aggregation == "ratio_of_means":
        score = float(ablated.mean() / full.mean())
    elif methodology.aggregation == "mean_of_ratios":
        score = float(torch.tensor(ratios).mean())
    else:
        span = reference.span
        score = float((ablated.mean() - reference.corrupted) / span) if span else float("nan")
    ordered = sorted(ratios)
    quarter = max(0, len(ordered) // 4)
    return Faithfulness(
        score=round(score, 4), methodology=methodology, per_example=[round(r, 4) for r in ratios],
        median=round(ordered[len(ordered) // 2], 4),
        iqr=(round(ordered[quarter], 4), round(ordered[-1 - quarter], 4)),
        worst=round(ordered[0], 4), best=round(ordered[-1], 4), n=len(ordered),
    )

def sensitivity(adapter, task: CircuitTask, circuit_heads: Sequence[HeadId],
                values: Sequence[str] = ABLATION_VALUES,
                directions: Sequence[str] = DIRECTIONS,
                sets: Sequence[str] = SETS,
                aggregations: Sequence[str] = AGGREGATIONS) -> List[Faithfulness]:
    """The same circuit under every combination, because the spread is the result

    This is the paper's experiment rather than its recommendation: run one
    circuit across the methodology axes and read how far the number travels. A
    circuit whose score is stable across the surface has earned a single
    number; one whose score doubles when mean becomes resample has not, and
    reporting either endpoint alone would be a choice presented as a finding.

    restore ignores the ablation value by construction -- it writes clean
    activations in, and there is no "off" to choose -- so those rows are
    computed once and the duplicates dropped rather than recomputed under
    labels that would suggest they differ.
    """
    reference = baselines(adapter, task)
    seen, results = set(), []
    for direction in directions:
        for ablated in sets:
            for value in values:
                for aggregation in aggregations:
                    effective = "n/a" if direction == "restore" else value
                    key = (direction, ablated, effective, aggregation)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(measure(
                        adapter, task, circuit_heads,
                        Methodology(value=value, direction=direction, ablated=ablated,
                                    aggregation=aggregation),
                        reference=reference,
                    ))
    return results

def report(results: Sequence[Faithfulness]) -> dict:
    """The surface as an artifact, with the spread stated rather than left to be noticed"""
    scores = [r.score for r in results]
    finite = [s for s in scores if s == s]
    return {
        "protocol": "faithfulness measured across the ablation-methodology axes of Miller, Chughtai and "
                    "Saunders (2407.08734). A score is reported only with the tuple that produced it; two "
                    "scores under different tuples are not comparable, which is that paper's finding.",
        "axes": {"value": list(ABLATION_VALUES), "direction": list(DIRECTIONS),
                 "ablated": list(SETS), "aggregation": list(AGGREGATIONS),
                 "granularity": "head", "component": "node", "positions": "all"},
        "n_measurements": len(results),
        "score_range": [round(min(finite), 4), round(max(finite), 4)] if finite else None,
        "spread": round(max(finite) - min(finite), 4) if finite else None,
        "measurements": [
            {**asdict(r.methodology), "score": r.score, "median": r.median,
             "iqr": list(r.iqr), "worst": r.worst, "best": r.best, "n": r.n}
            for r in results
        ],
    }
