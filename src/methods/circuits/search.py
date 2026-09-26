"""Growing a circuit: the set itself, and the greedy walk that finds one.

`patch_heads` scores every head alone. A circuit is a *set*, and a set of heads
that each score well individually is not the same thing as a set that works
together -- so the search restores whole sets and reads the curve.

The curve is the result, which is why the per-step scores are kept rather than
only the final one. A set whose recovery jumps at the third head and then
flattens is a circuit; one that climbs a little at every head is a model
spreading the task across everything, and the two are indistinguishable from
the threshold alone.

Greedy, and honest about it: this finds *a* sufficient set, not the smallest
one. `verify` is the module that catches what greed picks up, and `techniques`
is where the question of whether greed was the right search at all is asked --
this is one technique's answer, ranked by `patch_heads` and cut off at a
threshold, and the registry exists because the field stopped believing any
single one of those choices.

A common pipe could be: build_task | patch_heads | discover | verify
"""

from dataclasses import dataclass, field
from typing import List, Optional

from ...data.tasks import CircuitTask
from ...model.adapter import require_circuits
from ..common.components import HeadId
from ..common.errors import CircuitError
from .patching import HeadEffects, patch_heads, restore


@dataclass
class Circuit:
    """A set of heads, and the record of how they were chosen"""
    heads: List[HeadId]
    scores: List[float] = field(default_factory=list)
    threshold: float = 0.0

    def __len__(self) -> int:
        return len(self.heads)

    def __str__(self) -> str:
        names = ", ".join(f"L{layer}H{head}" for layer, head in self.heads)
        return f"{len(self.heads)} heads: {names}" if self.heads else "empty circuit"


def discover(
    adapter, dataset: CircuitTask, threshold: float = 0.8, max_heads: int = 12,
    effects: Optional[HeadEffects] = None,
) -> Circuit:
    """Grow a circuit greedily until restoring it alone reproduces the clean behaviour

    Candidates are tried in the order patch_heads ranked them, so the search is
    cheap rather than exhaustive, and it stops as soon as the set clears the
    threshold. The scores are kept per step because the shape of that curve is
    the result: a set whose recovery jumps at the third head and then flattens
    is a circuit, and one that climbs a little at every head is a model
    spreading the task across everything.

    Greedy means this finds *a* sufficient set, not the smallest one. verify's
    minimality column is what catches a passenger.
    """
    adapter = require_circuits(adapter)
    if max_heads < 1:
        raise CircuitError(f"a circuit needs room for at least one head, got max_heads={max_heads}")
    measured = effects or patch_heads(adapter, dataset)
    reference = measured.baselines
    donors = adapter.head_outputs(dataset.clean)

    chosen: List[HeadId] = []
    scores: List[float] = []
    for head_id, _ in measured.ranked(max_heads):
        chosen.append(head_id)
        scores.append(restore(adapter, dataset, reference, chosen, donors))
        if scores[-1] >= threshold:
            break
    return Circuit(heads=chosen, scores=scores, threshold=threshold)
