"""What a model does on a task before anything is taken away, and the span a corruption opened.

Every causal number in this layer is a fraction of one of two references, and
the choice between them is not a matter of taste:

    Baselines   clean and corrupted on the *same* prompts. A patched run is
                read as a recovery -- 0 at corrupted, 1 at clean -- so the
                span is the unit every restoration is quoted in.
    Behaviour   the clean run alone: how far apart the two answers are, and
                how often the right one wins. An ablation is read against this
                as damage, a share of the clean logit difference.

A recovery is a fraction of one task's corruption, which the second task in a
cross-task sweep does not have; damage is a fraction of the clean behaviour,
which every task has. That is why both live here rather than one being derived
from the other, and it is what makes "ablating IOI's circuit costs the
greater-than task 60% of its logit difference" a sentence with a meaning.

They sit in `common` because four things measure against them and none of them
owns the other: attribution checks its decomposition against the measured
difference, patching divides by the span, the technique registry compares
methods in recovery units, and the faithfulness surface re-measures the same
span under seven methodological axes. Before this module they all imported the
circuit study to get them.

A common pipe could be: task | baselines | patch | recovery
A common pipe could be: task | behaviour | ablate | damage
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from ...core.metrics import logit_difference, recovery
from ...data.tasks import CircuitTask
from ...model.adapter import require_circuits
from .errors import CircuitError


def mean_logit_difference(adapter, prompts: Sequence[str], io: Sequence[int], subject: Sequence[int]) -> float:
    """The mean logit difference over a batch of prompts"""
    return float(logit_difference(adapter.logits(list(prompts)), io, subject).mean())


@dataclass(frozen=True)
class Baselines:
    """The clean and corrupted behaviour a patched run is read against"""
    clean: float
    corrupted: float
    io: List[int]
    subject: List[int]

    @property
    def span(self) -> float:
        return self.clean - self.corrupted

    def recovery(self, patched: float) -> float:
        """How much of the span this patched run recovered, as 0 at corrupted and 1 at clean"""
        return recovery(patched, self.clean, self.corrupted)


def baselines(adapter, dataset: CircuitTask) -> Baselines:
    """Measure the clean and corrupted behaviour this dataset opens up

    A span near zero is fatal rather than merely disappointing: every later
    number divides by it, so a corruption that did not corrupt anything turns
    the whole study into noise amplified to look like a result.
    """
    adapter = require_circuits(adapter)
    io, subject = dataset.answers(adapter)
    clean = mean_logit_difference(adapter, dataset.clean, io, subject)
    corrupted = mean_logit_difference(adapter, dataset.corrupted, io, subject)
    found = Baselines(clean=clean, corrupted=corrupted, io=io, subject=subject)
    if abs(found.span) < 1e-6:
        raise CircuitError(
            f"the corruption moved the logit difference by {found.span:.2e}, so there is no span to recover; "
            "check that the corrupted prompts differ from the clean ones in the way you think"
        )
    return found


@dataclass(frozen=True)
class Behaviour:
    """What a model does on a task before anything is taken away from it"""
    logit_difference: float
    accuracy: float
    n: int


def behaviour(adapter, dataset: CircuitTask, prompts: Optional[Sequence[str]] = None) -> Behaviour:
    """Score the clean run: how far apart the two answers are, and how often the right one wins"""
    adapter = require_circuits(adapter)
    io, subject = dataset.answers(adapter)
    scores = logit_difference(adapter.logits(list(prompts if prompts is not None else dataset.clean)), io, subject)
    return Behaviour(
        logit_difference=float(scores.mean()),
        accuracy=float((scores > 0).to(torch.float64).mean()),
        n=len(io),
    )
