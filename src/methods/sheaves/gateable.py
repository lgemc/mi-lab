"""Which weights carry a gate, and how a band of them is named in a sentence.

DiscoGP attaches a gate to every weight in the blocks and to nothing else.
Norms and embeddings are excluded for a reason that is about the claim rather
than about arithmetic: a gated embedding deletes tokens rather than
computation, and a gated norm changes what every surviving component reads, so
neither is a statement about where the behaviour lives.

`layers` narrows that to a band, and *that* reason is arithmetic. Every gate
carries a float32 logit, its gradient and two AdamW moments, so gating all of a
1.7B model's block weights costs ~23 GiB of optimizer state before a single
activation is stored. A band spends less and says so -- the claim it supports
is "within these layers", which is a smaller claim and is written into the
artifact as one.

A band that the task survives without is the failure this cannot catch on its
own; `training.load_bearing` is the check that must run before the band is
trusted, because pruning inside an irrelevant band is free and reports a
density near zero at full accuracy.

A common pipe could be: adapter | gateable | load_bearing | prune
"""

from typing import Dict, Optional, Sequence

import torch

from ..common.errors import SheafError


def span(layers: Sequence[int]) -> str:
    """`21-27` for a contiguous band, the list itself for anything else"""
    ordered = sorted(layers)
    if len(ordered) > 1 and ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}-{ordered[-1]}"
    return ",".join(str(layer) for layer in ordered)

def gateable(adapter, layers: Optional[Sequence[int]] = None) -> Dict[str, torch.nn.Parameter]:
    """Every weight a gate is attached to: the blocks, minus norms and embeddings

    Norms and embeddings are excluded the way DiscoGP excludes them. A gated
    embedding deletes tokens rather than computation, and a gated norm changes
    what every surviving component reads, so neither is a statement about where
    the behaviour lives.

    `layers` narrows that to a band, and the reason is arithmetic rather than
    method. Every gate carries a float32 logit, its gradient and two AdamW
    moments, so gating all of a 1.7B model's 1.41B block weights costs ~23 GiB
    of optimizer state before a single activation is stored -- where GPT-2
    small's 85M gates cost 1.4 GiB, which is why this never came up. A band
    spends less and says so: the claim it supports is "within these layers",
    which is a smaller claim and is written into the artifact as one. Reaching
    for a leaner optimizer instead would buy the same memory by making the
    scope of the result harder to see rather than easier.
    """
    count = len(adapter.blocks)
    chosen = list(range(count)) if layers is None else list(layers)
    outside = [layer for layer in chosen if not 0 <= layer < count]
    if outside:
        raise SheafError(
            f"layers {outside} are not in this model, which has {count} blocks (0-{count - 1})"
        )
    if not chosen:
        raise SheafError("an empty layer band gates nothing; omit `layers` to gate the whole model")
    inside = {id(parameter) for layer in chosen
              for name, parameter in adapter.blocks[layer].named_parameters()
              if "ln" not in name and "norm" not in name}
    return {name: parameter for name, parameter in adapter.model.named_parameters()
            if id(parameter) in inside}
