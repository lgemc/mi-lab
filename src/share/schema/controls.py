from dataclasses import dataclass, field
from typing import List

from .control import Control

"""
The slate of checks a claim was tested against, written even when empty.

Both slots ship as empty lists rather than being left out, for the reason
`edges` does: an artifact that ran no cross-task ablation and one that ran
several are otherwise byte-identical, and no reader can tell "checked, and it
held" from "nobody thought to check".

`cross_task` is the one that matters most for a circuit. Ablating one task's
circuit damages unrelated tasks about as much as its own, because circuits at
this level are dominated by shared infrastructure -- so within-task
faithfulness is consistent with the circuit being general-purpose machinery,
and a circuit reporting only within-task numbers is not shown to be about its
task at all.

A common pipe could be: Control | Controls | artifact check
"""

@dataclass(frozen=True)
class Controls:
    """The checks a claim was tested against, listed even when nothing was run"""
    cross_task: List[Control] = field(default_factory=list)
    random_baseline: List[Control] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        """Whether nothing at all was run, which is a thing a reader must be told"""
        return not self.cross_task and not self.random_baseline
