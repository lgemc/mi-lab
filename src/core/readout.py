from typing import Any, Protocol, Sequence, runtime_checkable

import torch

from .metrics import MetricError

"""
One number per example, in stated units, with the definition that produced it.

Everything in `methods` consumes exactly three things: a batch of inputs, a set
of addressable internal sites, and one scalar per example. Two of the three were
already abstract. The third was a logit difference over two token ids, hardcoded
in `methods/common/span.py` and carried into every technique as a pair of
`List[int]`, and that single fact is what made twenty-three otherwise
modality-free modules into language modules.

This is the third thing, named. A task hands back the scoring rule for the model
in hand, the same way it already handed back answer ids -- and for the same
reason, written in `CircuitTask.answers`'s own docstring: the ids are the
tokenizer's opinion and a task carrying them would be silently wrong on the next
model. The generalization is one step further. *The whole scoring rule* is the
task's opinion about this model, not just the ids inside it.

It lives in `core` rather than in `methods/common/` -- where the proposal put it
-- because `data/tasks.py` has to name the type `CircuitTask.readout` returns,
and `data` sits below `methods`. Core's charter is what a model is and how a
number is scored, and a Score is the second of those with the arithmetic taken
out, so it earns the place `core/metrics.py` holds beside it.

`definition` is not documentation. `share/schema/metric.py` refuses a Metric
with an empty definition, and `share/definitions.py` is a lookup table keyed by
metric *name* that can disagree with the code that produced the number -- the
same class of bug as a `.pt` file identified by its filename. A Score carries
its own, so an artifact's definition is written by the thing that computed it.

A common pipe could be: task | readout | baselines | recovery
"""


class ReadoutError(MetricError):
    """Raised when a score cannot say what it measures, or cannot be differentiated through"""


@runtime_checkable
class Score(Protocol):
    """One number per example, higher meaning more of the behaviour

    `select` is what chunking needs and a single call does not. Every backend
    here runs a batch in chunks of `batch_size`, and a score built for the whole
    batch -- one answer id per row -- has to be narrowed to the rows of the chunk
    it is scoring. That is the same reason `_Patch.select` exists in the patching
    backend: a hook or a score that closes over the whole batch scores whichever
    chunk happened to run last. A score with no per-example state is its own
    restriction and returns `self`.
    """

    name: str
    units: str
    definition: str
    #: Whether `__call__` keeps the autograd graph. See `Readout` for why this
    #: is a field rather than a base class.
    differentiable: bool

    def __call__(self, output: Any) -> torch.Tensor:
        """[batch] floats, one per example, higher meaning more of the behaviour"""

    def select(self, rows: slice) -> "Score":
        """The same score restricted to a slice of the batch"""


@runtime_checkable
class Readout(Score, Protocol):
    """A Score that a gradient can be taken through

    What it promises is that `__call__` is a differentiable function of `output`
    with the graph intact -- which is what eap, eap_ig and mask training need and
    what BLEU cannot give.

    It adds no members, and that is exactly why `differentiable` is a field on
    `Score` instead. A Protocol with no members of its own is satisfied
    structurally by *everything*: `isinstance(bleu, Readout)` would be True and
    the refusal in `require_readout` would never fire. So the marker is data the
    score declares about itself, this class is the name for it in a signature,
    and `require_readout` is the only thing that decides.

    A score that declares itself differentiable and detaches is a bug in the
    score. It is caught by `tests/methods/discovery.py`'s finite difference and
    by the `grad_fn` check every readout's own test makes, not by a type.
    """


@runtime_checkable
class DirectScore(Score, Protocol):
    """A Score that can also be applied to one write into the residual stream

    Direct attribution does not score an output; it scores each component's
    contribution *to* that output, through a frozen final norm. That needs the
    score to know what the write is dotted with -- for a logit difference, the
    two unembedding rows -- which is domain knowledge the decomposition itself
    does not carry.

    A score that has no such form is not thereby a bad score. It just cannot be
    decomposed, and `require_direct` says so by name rather than failing on a
    missing attribute three frames in.
    """

    def direct(self, unembedding, writes: torch.Tensor) -> torch.Tensor:
        """Score a [batch, ..., d_model] stack of residual writes, as [batch, ...]"""

    def unattributed(self, unembedding) -> torch.Tensor:
        """The part of the score that belongs to no component at all, as [batch]"""


def require_score(score) -> Score:
    """Reject a score that cannot say what it measures

    An empty `units` is refused for the reason `Metric` refuses an empty
    `definition`: `viz` reads `score.units` for an axis label, so a blank one is
    an unlabelled chart rather than an error, and an unlabelled chart is read as
    if it were labelled.
    """
    if not isinstance(score, Score):
        raise ReadoutError(
            f"{type(score).__name__} is not a Score: it needs name, units, definition, differentiable, "
            "__call__ and select"
        )
    for field in ("name", "units", "definition"):
        if not getattr(score, field, "").strip():
            raise ReadoutError(
                f"the score '{getattr(score, 'name', '?')}' ships an empty {field}; a number whose "
                f"{field} nobody recorded cannot be read by the next person to hold it"
            )
    return score


def require_readout(score, by: str = "") -> Readout:
    """Reject a score no gradient can be taken through, before the model loads"""
    if not getattr(score, "differentiable", False):
        asked = f"'{by}' needs" if by else "this needs"
        raise ReadoutError(
            f"{asked} a gradient through the score, and the readout '{score.name}' has none. "
            "Score it with a technique that only runs the model forward, or give the task a "
            "differentiable readout."
        )
    return score


def require_direct(score, by: str = "") -> DirectScore:
    """Reject a score that cannot be split over the writes that produced it"""
    if not isinstance(score, DirectScore):
        asked = f"'{by}' splits" if by else "this splits"
        raise ReadoutError(
            f"{asked} a score over each component's write into the residual stream, and the readout "
            f"'{score.name}' has no direct form. Only a readout that is linear in the final residual "
            "stream can be decomposed that way."
        )
    return score


def mean_score(adapter, inputs: Sequence[Any], score: Score) -> float:
    """The mean of this score over a batch, which is what a baseline is"""
    return float(score(adapter.outputs(list(inputs))).mean())
