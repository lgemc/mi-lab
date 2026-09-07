from dataclasses import dataclass
from typing import List, Sequence

import torch

from ...core.metrics import logit_difference
from ...core.readout import ReadoutError

"""
How a number comes out of a decoder: the difference between two token logits.

This is the readout `methods/common/span.py` used to *be*. It sat there as a
hardcoded call plus a pair of `List[int]` carried on `Baselines` into every
technique downstream, and that pair was what pinned twenty-three modality-free
modules to language. Here it is one object with a name, units and a definition,
built against the model in hand by the task that knows which two tokens the
question is between.

Only differences of two logits, never a logit on its own -- the same rule
`Unembedding` states and for the same reason: a single logit carries the
softmax's arbitrary additive constant, so every component appears to move it,
while the difference between two tokens is the quantity the behaviour is made
of.

A common pipe could be: task | readout | baselines | eap
"""

#: What the number is, written where the number is computed. `share/definitions.py`
#: keys the same sentence by metric name and is now the fallback for migrating old
#: artifacts rather than the source for new ones -- a table keyed by name is a
#: second opinion about what was measured, and nothing checked that it agreed.
DEFINITION = (
    "logit(answer) - logit(distractor) at each prompt's final real token, one token pair per "
    "example; a difference rather than a probability, so the softmax normalizer the two answers "
    "share cancels"
)


@dataclass(frozen=True)
class LogitDifference:
    """The two-answer readout, bound to one model's token ids

    Bound, and therefore not to be held across a model swap. The ids are the
    tokenizer's opinion: the same object used on another checkpoint scores every
    prompt against whatever those ids happen to mean there, silently. That is
    `Means.check`'s failure mode, so it gets `Means.check`'s answer -- the config
    id is stamped in and `check` refuses another.
    """

    positive: List[int]
    negative: List[int]
    model: str = ""

    name = "logit_difference"
    units = "logits"
    definition = DEFINITION
    differentiable = True

    def __post_init__(self) -> None:
        if len(self.positive) != len(self.negative):
            raise ReadoutError(
                f"{len(self.positive)} positive and {len(self.negative)} negative token ids; "
                "they index the same batch, one pair per example"
            )
        if not self.positive:
            raise ReadoutError("a logit difference needs at least one answer pair to score")

    def __call__(self, output: torch.Tensor) -> torch.Tensor:
        """logit(positive) - logit(negative) per row, given [batch, vocab] next-token logits

        One implementation, used for the number that gets reported *and* for the
        objective a gradient is taken through. It was two before -- the reporting
        path went through `core.metrics.logit_difference` and the backend's
        gradient path indexed the logits itself -- and two implementations of one
        equation is two chances to differentiate something other than what was
        measured.
        """
        return logit_difference(output, self.positive, self.negative)

    def select(self, rows: slice) -> "LogitDifference":
        """The same readout over a slice of the batch, for a backend scoring one chunk"""
        return LogitDifference(
            positive=list(self.positive[rows]), negative=list(self.negative[rows]), model=self.model
        )

    def direct(self, unembedding, writes: torch.Tensor) -> torch.Tensor:
        """What each residual-stream write was directly worth, through the frozen final norm

        The decomposition is unchanged by this contract; only what the final
        vector is dotted with moves from a fixed pair of unembedding rows to the
        pair this readout holds. `Attribution.residual` is the receipt that it is
        still the same quantity, and it closes at ~1e-6 on GPT-2 small.
        """
        return unembedding.logit_difference(writes, self.positive, self.negative)

    def unattributed(self, unembedding) -> torch.Tensor:
        """The final norm's own shift, which lands on the answer without any component writing it"""
        return unembedding.offset(self.positive, self.negative)

    def check(self, adapter) -> "LogitDifference":
        """Refuse a model this readout was not built against

        `telemetry/results.py::guard` and `methods/knockout/ablate.py::Means` are
        the two precedents in this repository and they agree: a cached thing that
        carries a model's opinion states which model, and refuses another rather
        than answering a different question in the same units.
        """
        if self.model and self.model != adapter.cfg.id:
            raise ReadoutError(
                f"this readout's token ids were read off '{self.model}' and it is being scored on "
                f"'{adapter.cfg.id}'; ask the task for a readout built against the model in hand"
            )
        return self


def logit_difference_readout(adapter, positive: Sequence[int], negative: Sequence[int]) -> LogitDifference:
    """The two-answer readout for the model in hand, stamped with which model that was"""
    return LogitDifference(positive=list(positive), negative=list(negative), model=adapter.cfg.id)
