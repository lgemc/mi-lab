"""Taking a set of heads away from the clean run, and reading the loss as damage rather than recovery.

Patching restores into a corrupted run and asks how much came back. Ablation
is the same intervention from the other side: take the heads out of the clean
run and ask what the task lost. The two come apart wherever the model has a
second route to the answer, and that is the point of having both.

The unit is the whole reason this is not a function in `patching`. A recovery
is a fraction of one task's corruption span, so it is undefined for a second
task whose corruption is a different operation on different prompts. Damage is
a fraction of the clean behaviour itself, which every task has -- so "ablating
IOI's circuit costs the greater-than task 60% of its logit difference" is a
sentence with a meaning, and the cross-task sweep in `comparison` is built out
of it.

The trap is in `donor_bank` and it is worth reading before quoting a diagonal:
a mean ablation only removes what *varied across the batch*, and every task in
`data/tasks.py` is built from one frame. On a task whose prompts differ in one
slot the mean is close to every row, so the ablation is close to a no-op, and a
near-zero diagonal is the ablation rather than the model.

A common pipe could be: build_task | donor_bank | ablate | damage
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from ...data.tasks import CircuitTask
from ...model.adapter import require_circuits
from ..common.components import HeadId
from ..common.errors import CircuitError
from ..common.intervention import head_patch
from ..common.span import Behaviour, behaviour

ABLATIONS = ("mean", "corrupted")


@dataclass(frozen=True)
class Ablation:
    """What a task's behaviour looks like once a set of heads is taken away

    This is the number a cross-task claim is made of, and it is deliberately
    not a recovery. A recovery is a fraction of one task's corruption span, so
    it is undefined for a second task whose corruption is a different
    operation on different prompts; damage is a fraction of the clean
    behaviour itself, which every task has. That is what makes "ablating IOI's
    circuit costs the greater-than task 60% of its logit difference" a
    sentence with a meaning.
    """
    heads: List[HeadId]
    clean: float
    ablated: float
    accuracy: float
    clean_accuracy: float
    donor: str

    @property
    def damage(self) -> float:
        """Share of the clean logit difference this ablation removed, 1.0 for all of it"""
        if self.clean == 0:
            return 0.0
        return (self.clean - self.ablated) / self.clean

    def __str__(self) -> str:
        return (
            f"{len(self.heads)} heads {self.donor}-ablated: logit difference {self.clean:+.3f} -> "
            f"{self.ablated:+.3f} (damage {self.damage:.0%}), accuracy {self.clean_accuracy:.0%} -> "
            f"{self.accuracy:.0%}"
        )


def donor_bank(adapter, dataset: CircuitTask, donor: str) -> torch.Tensor:
    """The activations an ablation writes in, as a full [batch, layer, head, seq, d_head] bank

    'mean' averages each head's output over the task's own clean prompts and
    hands every row the same value, which removes what the head *knew about
    this example* while leaving the model in the distribution it was measured
    in. 'corrupted' writes the twin run's values, which is the same operation
    verify's necessity column performs and is only defined when the second
    task's corruption is the one being asked about.

    What a mean ablation removes is only what *varied across the batch*, and
    every task here is built from one frame -- so on a task whose prompts
    differ in one slot, the mean is close to every row and the ablation is
    close to a no-op. A diagonal near zero in a cross-task sweep is that, as
    often as it is a task the attention heads do not carry.
    """
    if donor not in ABLATIONS:
        raise CircuitError(f"unknown ablation donor '{donor}'; known donors are {sorted(ABLATIONS)}")
    if donor == "corrupted":
        return adapter.head_outputs(dataset.corrupted)
    clean = adapter.head_outputs(dataset.clean)
    return clean.mean(dim=0, keepdim=True).expand_as(clean).contiguous()


def ablate(
    adapter,
    dataset: CircuitTask,
    heads: Sequence[HeadId],
    donor: str = "mean",
    donors: Optional[torch.Tensor] = None,
    clean: Optional[Behaviour] = None,
) -> Ablation:
    """Take a set of heads out of the clean run and see what the task loses

    `donors` and `clean` are here so that a sweep over many head sets pays for
    the donor bank and the clean baseline once. Passing a bank measured on a
    different task is the mistake they make possible, and it is the reason
    both are keyword arguments with honest defaults rather than a cache.
    """
    adapter = require_circuits(adapter)
    chosen = [(int(layer), int(head)) for layer, head in heads]
    reference = clean if clean is not None else behaviour(adapter, dataset)
    if not chosen:
        return Ablation(
            heads=[], clean=reference.score, ablated=reference.score,
            accuracy=reference.accuracy, clean_accuracy=reference.accuracy, donor=donor,
        )

    bank = donors if donors is not None else donor_bank(adapter, dataset, donor)
    with adapter.patch(heads=head_patch(chosen, bank)):
        after = behaviour(adapter, dataset)
    return Ablation(
        heads=chosen, clean=reference.score, ablated=after.score,
        accuracy=after.accuracy, clean_accuracy=reference.accuracy, donor=donor,
    )
