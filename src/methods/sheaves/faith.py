from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import torch
from torch.nn import functional

from ..common.errors import SheafError

"""
What "the masked model still says what the model said" means, and the three
things it has meant here.

`--faith {pair,kl,nll,gold}` was a string threaded through four call sites and
compared against tuples of literals in three of them, and each of the three
lessons this repository paid for on the 1.7B was a comment beside one of those
comparisons. They are `Score`s: each carries the sentence that says what it
measures and when it lies, and the flag now selects an object rather than
steering a branch.

The lessons, kept with the kinds they belong to:

    nll    the paper's term, and the one that collapses. It is the likelihood
           the masked model gives the token the *full model* predicted, which
           on a frame like `The Spanish word X means` is ` "` for 274 of 300
           pool words. Every run on that frame learned to emit quotes at
           95-98% density while faith read ~0.
    kl     the whole distribution rather than its argmax. The paper evaluates
           with it, and it is what to train with on a frame whose next token
           is not already the answer.
    gold   nll against the *task's* answer rather than the full model's argmax.
           Not the paper's faithfulness, and it is the quantity the probes
           score, so it is the one to reach for when the question is whether
           the circuit does the task rather than whether it imitates the model.
    pair   this repo's original: a two-way comparison between the answer and
           one distractor. Too weak -- two logits can both be beaten by
           everything else in the distribution -- and kept because runs exist
           that were trained under it.

Every one of them is differentiable, which is the whole reason a mask can be
trained at all; `TextQuality` in `methods/knockout/quality.py` is the
counterexample the `Score`/`Readout` split exists for.

A common pipe could be: faith_for | logit_pairs | term | backward
"""


@dataclass(frozen=True)
class Faithfulness:
    """One meaning of faithfulness, as a Score over a batch's final logits

    `whole` is the shape it needs from `logit_pairs`: the two kinds that
    compare against the full model's own distribution need every logit, and
    `pair` needs two. Reading it off the object is what stops the three call
    sites that used to write `faith_kind in ("kl", "nll", "gold")` from
    disagreeing the day a fourth kind arrives.
    """

    name: str
    units: str
    definition: str
    whole: bool
    term: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    differentiable = True

    def __call__(self, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """The loss for one batch, against the reference this kind wants"""
        return self.term(output, labels)

    def select(self, rows: slice) -> "Faithfulness":
        """A faithfulness term holds no per-example state; the labels are passed in"""
        return self


def _cross_entropy(pairs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return functional.cross_entropy(pairs, labels)


def _kl(pairs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return functional.kl_div(pairs.log_softmax(dim=-1), labels, log_target=True, reduction="batchmean")


def _prefer_answer(pairs: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
    # the answer is column 0 of every pair, so the label is the same for every row
    return functional.cross_entropy(
        pairs, torch.zeros(pairs.shape[0], dtype=torch.long, device=pairs.device)
    )


FAITH: Dict[str, Faithfulness] = {
    "nll": Faithfulness(
        name="nll",
        units="nats",
        definition=(
            "-sum_i log p_masked(y-hat_i | x_i) over the whole vocabulary, where y-hat is the token "
            "the *full* model predicted; the paper's faithfulness term, and the one that collapses "
            "on a frame whose argmax is punctuation rather than the answer"
        ),
        whole=True,
        term=_cross_entropy,
    ),
    "kl": Faithfulness(
        name="kl",
        units="nats",
        definition=(
            "KL(full || masked) over the whole next-token distribution, the soft-target relative of "
            "nll; the quantity the paper evaluates with"
        ),
        whole=True,
        term=_kl,
    ),
    "gold": Faithfulness(
        name="gold",
        units="nats",
        definition=(
            "-sum_i log p_masked(y_i | x_i) over the whole vocabulary, where y is the *task's* "
            "answer; not the paper's faithfulness, and the quantity the probes score"
        ),
        whole=True,
        term=_cross_entropy,
    ),
    "pair": Faithfulness(
        name="pair",
        units="nats",
        definition=(
            "cross entropy between the answer's and the distractor's logits alone; too weak, since "
            "two logits can satisfy it while everything else in the distribution outranks both"
        ),
        whole=False,
        term=_prefer_answer,
    ),
}


def faith_kinds() -> Sequence[str]:
    """Every meaning of faithfulness this module knows, sorted"""
    return sorted(FAITH)


def faith_for(kind: str) -> Faithfulness:
    """The faithfulness term a flag names, or a refusal listing the ones that exist"""
    if kind not in FAITH:
        raise SheafError(
            f"unknown faith_kind '{kind}'; known kinds are 'nll' (the paper's), 'kl', "
            "'gold' (nll on the task's answer) and 'pair' (this repo's original, and too "
            "weak -- see faith.py)"
        )
    return FAITH[kind]
