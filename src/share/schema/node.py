from dataclasses import dataclass, field
from typing import Dict, Optional

from .vocabulary import NodeComponent

"""
One component of a circuit, holding every measurement made of it.

`scores` is an open map rather than a pair of fields, because the circuit
question is asked twice by methods that see different things: attribution
reads the direct path only and is exact, patching is causal and expensive.
They disagree, and the disagreement is the finding -- on GPT-2 small the
negative name movers write hard against the answer and patching says the model
needs them. One summary score per component throws that away.

A common pipe could be: discover | verify | Node | Artifact
"""

@dataclass(frozen=True)
class Node:
    """One component of a circuit, with what both halves of the study said about it

    `role` is prose and stays prose: it is a name somebody gave a behaviour,
    and the set grows as the task does. The ones this repository assigns are
    "name mover", "s-inhibition", "duplicate token" and "induction"; None means
    no role was assigned rather than that the head has none.

    `scores` holds every measurement made of this node, keyed by what it
    measured. It is an open map for the same reason -- a new measurement should
    not need a schema change before it can be shared. The keys this repository
    writes are `attribution` (direct-path logit contribution, in logits),
    `causal` (patching recovery, a fraction of the span), `minimality` (the
    faithfulness lost when this head alone is dropped), `cumulative_recovery`
    (what the circuit reached once this head was added) and `step` (where in
    the greedy search that happened). `attribution` and `causal` are both kept
    because they disagree, and a circuit format that stores only one of them
    throws away the finding.
    """
    id: str
    component: NodeComponent
    layer: int
    head: Optional[int] = None
    role: Optional[str] = None
    in_circuit: bool = False
    scores: Dict[str, float] = field(default_factory=dict)
