"""Putting something into a model and taking it out again, which this repo does four ways.

Every causal claim here is made by changing the model, running it, and putting
it back. Four places do that and none of them knew about the others:

    circuits.ablation   writes a donor activation over a head's output
    knockout.ablate     writes a counterfactual mean over a head or an MLP,
                        under hooks, on a generation
    sheaves.mask        multiplies a trained bit mask into the weights
    serve.backbones     wraps either of the last two for a request

They differ in what they write and in what it costs -- a weight mask is paid
once and then decodes at full speed, an edge gate is paid on every token -- and
they agree exactly on the shape: a context manager that leaves the model as it
found it, whether or not the body raised. `Intervention` writes that agreement
down. It is a `Protocol`, so nothing has to inherit it and the four keep their
own machinery; what it buys is that a function taking "some way of changing the
model" can say so in its signature, and that a new kind has a contract to meet
rather than three examples to imitate.

`describe` is in the protocol and not an afterthought. An intervention that
cannot say what it did produces a number nobody can attribute, which is the
same failure `Payload` refuses for a tensor and `Metric` for a float.

`head_patch` is the other half: the nested dict `adapter.patch` takes. It was
written out by hand at seven call sites across `circuits`, `comparison` and
`faithfulness`, every one of them the same three lines, and the interesting
part -- which activation goes in -- was buried in the middle of them.

A common pipe could be: head_patch | adapter.patch | measure
"""

from typing import Callable, ContextManager, Dict, Protocol, Sequence, runtime_checkable

import torch

from .components import HeadId


@runtime_checkable
class Intervention(Protocol):
    """Something that can be put into a model and taken out again, leaving no trace

    The contract is the restoration, not the change. An implementation that
    can enter but cannot reliably leave is worse than none: the next
    measurement runs on a half-masked model and reports a number that looks
    ordinary. So `applied` restores in a `finally`, and it restores *exactly*
    -- a weight put back approximately is a checkpoint nobody can compare
    against the one it came from.
    """

    def applied(self) -> ContextManager[None]:
        """Enter the intervention, and undo it on the way out however the body ends"""
        ...

    def describe(self) -> Dict[str, object]:
        """What was changed, in terms a result can carry: what kind, how much, of what"""
        ...


def head_patch(
    heads: Sequence[HeadId],
    source: torch.Tensor | Callable[[int, int], torch.Tensor],
) -> Dict[int, Dict[int, torch.Tensor]]:
    """The `{layer: {head: activation}}` that `adapter.patch` takes, for one set of heads

    `source` is normally a full `[batch, layer, head, seq, d_head]` bank, which
    is why the helpers that use it never take a layer subset: a layer index is
    a row index into the bank, and a bank measured over layers 9-11 would
    quietly answer for layers 0-2. Pass a callable instead when the value
    depends on the head rather than only on where it is -- the faithfulness
    surface picks between zero, the batch mean and the counterfactual run per
    head, and that choice is the measurement rather than a detail of it.

    Nothing here decides *which* activation is right. Writing back what was
    already there is exactly a no-op, and that is the property every causal
    number in this layer is a difference against; handing this function a bank
    from another task is how that property is lost, and it is why the bank is
    always an argument rather than a cache.
    """
    pick = source if callable(source) else (lambda layer, head: source[:, layer, head])
    patch: Dict[int, Dict[int, torch.Tensor]] = {}
    for layer, head in heads:
        patch.setdefault(int(layer), {})[int(head)] = pick(int(layer), int(head))
    return patch
