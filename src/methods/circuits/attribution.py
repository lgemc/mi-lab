"""The correlational half of a circuit study: what each component wrote towards the answer.

The residual stream is a sum, so the logit difference the model ends on is the
sum of what every head, every MLP and the embedding wrote into it. That
decomposition is exact and it is cheap -- one forward pass answers for every
component at once -- and it only ever sees the *direct* path. A head that
matters by changing what a later head reads contributes nothing to it, and a
head that writes the answer while a later head deletes it looks like a hero.

So this is where to look, not what is true. `patching` is the half that
settles things, and the two disagree: on GPT-2 small the late negative name
movers have large negative attribution here and patching says the model needs
them. The disagreement is the finding rather than a bug in either.

`residual` is the receipt. Every write into the stream, summed and pushed
through the frozen unembedding, has to land on the logit difference the model
actually produced; on a model whose layout the adapter understands it is
numerically zero, and `tests/methods/circuits.py` holds it to 1e-6. A decomposition
that does not add up is one where a component's write is being read at the
wrong site, which produces plausible numbers and a wrong ranking.

A common pipe could be: build_task | direct_logit_attribution | top | patch_heads
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch

from ...core.readout import mean_score, require_direct
from ...data.tasks import CircuitTask
from ...model.adapter import require_circuits
from ..common.components import HeadId


@dataclass
class Attribution:
    """What each component wrote towards the answer, in logits, averaged over the batch"""
    heads: torch.Tensor
    mlps: torch.Tensor
    embedding: float
    offset: float
    measured: float

    @property
    def total(self) -> float:
        """The attribution summed back up, which must land on the measured logit difference"""
        return float(self.heads.sum() + self.mlps.sum()) + self.embedding + self.offset

    @property
    def residual(self) -> float:
        """Measured minus attributed: the check that the decomposition is complete"""
        return self.measured - self.total

    def top(self, count: int = 10, negative: bool = False) -> List[Tuple[HeadId, float]]:
        """The heads pushing hardest towards the answer, or away from it"""
        flat = self.heads.flatten()
        order = torch.argsort(flat, descending=not negative)[:count]
        width = self.heads.shape[1]
        return [((int(index) // width, int(index) % width), float(flat[index])) for index in order]


def direct_logit_attribution(adapter, dataset: CircuitTask, corrupted: bool = False) -> Attribution:
    """Split the logit difference into what each head and MLP directly contributed

    One forward pass answers for every component, because the residual stream
    is a sum and the unembedding is linear once the final norm's divisor is
    frozen. `residual` is the receipt: it is the part of the measured logit
    difference this split failed to account for, and on a model whose layout
    the adapter understands it is numerically zero.

    Read the result as a correlation, not a cause. It says what a component
    wrote towards the answer along the direct path to the unembedding, and
    says nothing about a component whose whole job is to change what a later
    one reads.
    """
    adapter = require_circuits(adapter)
    prompts = dataset.corrupted if corrupted else dataset.clean
    # a readout with no direct form is not a bad readout, it is one that cannot be
    # split over the writes that produced it; the refusal names it rather than
    # failing on a missing attribute inside the sum below
    score = require_direct(dataset.readout(adapter), by="direct attribution")
    decomposition = adapter.decompose(prompts)
    unembedding = decomposition.unembedding
    return Attribution(
        heads=score.direct(unembedding, decomposition.heads).mean(dim=0),
        mlps=score.direct(unembedding, decomposition.mlps).mean(dim=0),
        embedding=float(score.direct(unembedding, decomposition.embedding).mean()),
        offset=float(
            score.direct(unembedding, decomposition.biases).sum(dim=1).mean()
            + score.unattributed(unembedding).mean()
        ),
        measured=mean_score(adapter, prompts, score),
    )
