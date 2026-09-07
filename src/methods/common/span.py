"""What a model does on a task before anything is taken away, and the span a corruption opened.

Every causal number in this layer is a fraction of one of two references, and
the choice between them is not a matter of taste:

    Baselines   clean and corrupted on the *same* inputs. A patched run is
                read as a recovery -- 0 at corrupted, 1 at clean -- so the
                span is the unit every restoration is quoted in.
    Behaviour   the clean run alone: how far apart the answer and its
                alternative are, and how often the right one wins. An ablation
                is read against this as damage, a share of the clean score.

A recovery is a fraction of one task's corruption, which the second task in a
cross-task sweep does not have; damage is a fraction of the clean behaviour,
which every task has. That is why both live here rather than one being derived
from the other, and it is what makes "ablating IOI's circuit costs the
greater-than task 60% of its score" a sentence with a meaning.

They sit in `common` because four things measure against them and none of them
owns the other: attribution checks its decomposition against the measured
difference, patching divides by the span, the technique registry compares
methods in recovery units, and the faithfulness surface re-measures the same
span under seven methodological axes. Before this module they all imported the
circuit study to get them.

Neither of them knows what the number is. `Baselines` carried `io` and
`subject` -- two lists of token ids -- into every downstream consumer, and
`baselines` called a hardcoded `mean_logit_difference`; that pair was the
single fact that made this layer a language layer, since every technique
needing a gradient received the two lists and handed them to the backend.
What sits there now is the `Score` the task built for the model in hand, and
`Baselines.span` and `Baselines.recovery` are the same arithmetic they always
were, because arithmetic on a scalar never cared which scalar it was.

A common pipe could be: task | baselines | patch | recovery
A common pipe could be: task | behaviour | ablate | damage
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from ...core.metrics import recovery
from ...core.readout import Score, mean_score, require_score
from ...data.tasks import CircuitTask
from ...model.adapter import require_circuits
from .errors import CircuitError


@dataclass(frozen=True)
class Baselines:
    """The clean and corrupted behaviour a patched run is read against"""
    clean: float
    corrupted: float
    readout: Score

    @property
    def span(self) -> float:
        return self.clean - self.corrupted

    @property
    def units(self) -> str:
        """What `clean`, `corrupted` and `span` are in, according to the thing that measured them"""
        return self.readout.units

    def recovery(self, patched: float) -> float:
        """How much of the span this patched run recovered, as 0 at corrupted and 1 at clean"""
        return recovery(patched, self.clean, self.corrupted)


def baselines(adapter, dataset: CircuitTask) -> Baselines:
    """Measure the clean and corrupted behaviour this dataset opens up

    A span near zero is fatal rather than merely disappointing: every later
    number divides by it, so a corruption that did not corrupt anything turns
    the whole study into noise amplified to look like a result. The refusal can
    now name the readout that failed to move, rather than saying "logit
    difference" on a task whose number is something else.
    """
    adapter = require_circuits(adapter)
    score = require_score(dataset.readout(adapter))
    clean = mean_score(adapter, dataset.clean, score)
    corrupted = mean_score(adapter, dataset.corrupted, score)
    found = Baselines(clean=clean, corrupted=corrupted, readout=score)
    if abs(found.span) < 1e-6:
        raise CircuitError(
            f"the corruption moved '{score.name}' by {found.span:.2e} {score.units}, so there is no span "
            "to recover; check that the corrupted examples differ from the clean ones in the way you think"
        )
    return found


@dataclass(frozen=True)
class Behaviour:
    """What a model does on a task before anything is taken away from it"""
    score: float
    units: str
    accuracy: float
    n: int


def behaviour(adapter, dataset: CircuitTask, prompts: Optional[Sequence[Any]] = None) -> Behaviour:
    """Score the clean run: how far the readout sits from zero, and how often it is positive

    `accuracy` is the share of examples the readout scores on the correct side
    of zero, which is what "the model got it right" means for any score written
    so that more of the behaviour is a larger number. That convention is the
    `Score` protocol's one requirement on the sign, and it is what lets this
    stay one function rather than one per task.
    """
    adapter = require_circuits(adapter)
    reading = require_score(dataset.readout(adapter))
    scores = reading(adapter.outputs(list(prompts if prompts is not None else dataset.clean)))
    return Behaviour(
        score=float(scores.mean()),
        units=reading.units,
        accuracy=float((scores > 0).to(torch.float64).mean()),
        n=int(scores.shape[0]),
    )
