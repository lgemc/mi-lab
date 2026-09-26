"""The one module here that is about indirect object identification in particular.

Everything else in this package takes a `CircuitTask` -- clean prompts,
corrupted twins, two answer ids -- and runs on any of the four tasks in
`data/tasks.py`. This one names the four attention movements IOI is built out
of, so it takes an `IOIDataset` and says so in its signature rather than
failing on the third line of a greater-than sweep.

It is also the only measurement here that reads attention weights instead of
activations, and that is a weaker instrument on purpose: attention is where a
head looked, not what it did with what it found. A head can attend hard to the
indirect object and write nothing, and a head can matter through a path that
never looks at the answer at all. So this names candidates and `patching`
decides, which is the same division of labour as `attribution` and for the
same reason.

The threshold is what keeps the naming honest. Every head's strongest role is
*some* role, so without a floor every head in the model gets a label and the
classification says nothing.

A common pipe could be: build_ioi | classify_heads | assign | patch_heads
"""

from dataclasses import dataclass
from typing import Dict, Sequence

import torch

from ....methods.common.components import HeadId
from ....model.adapter import require_circuits
from ..data.ioi import IOIDataset

ROLES = ("name mover", "s-inhibition", "duplicate token", "induction")


@dataclass
class HeadRoles:
    """How much attention each head pays to the four movements the task is made of"""
    weights: torch.Tensor
    roles: Sequence[str] = ROLES

    def assign(self, threshold: float = 0.3) -> Dict[HeadId, str]:
        """Name each head after the movement it spends most of its attention on

        The threshold is what keeps the classification honest. Every head's
        strongest role is *some* role, and a head paying five percent of its
        attention to the indirect object is not a name mover; below the
        threshold a head simply gets no name.
        """
        best, index = self.weights.max(dim=-1)
        assigned = {}
        for layer in range(self.weights.shape[0]):
            for head in range(self.weights.shape[1]):
                if float(best[layer, head]) >= threshold:
                    assigned[(layer, head)] = self.roles[int(index[layer, head])]
        return assigned


def classify_heads(adapter, dataset: IOIDataset) -> HeadRoles:
    """Score every head on the four attention movements IOI is built out of

    Each role is a query position and a key position, because a head's job is
    where it looks *from* as much as where it looks *to*:

    - name mover: from the end of the sentence to the indirect object -- the
      head that fetches the answer.
    - s-inhibition: from the end to the repeated name, which is how the
      end-of-sentence position learns which name is already spoken for.
    - duplicate token: from the second mention of the subject back to the
      first -- the head that notices the repetition at all.
    - induction: from the second mention to the token *after* the first, the
      generic copy machinery the duplicate signal can also ride on.

    Attention is where a head looked, not what it did with it. This names
    candidates; patch_heads decides.
    """
    adapter = require_circuits(adapter)
    landmarks = dataset.landmarks(adapter)
    end, io, subject_first, subject_second = (landmarks[key] for key in ("END", "IO", "S1", "S2"))
    patterns = adapter.attention(dataset.clean)

    queries_keys = (
        (end, io),
        (end, subject_second),
        (subject_second, subject_first),
        (subject_second, min(subject_first + 1, patterns.shape[-1] - 1)),
    )
    weights = torch.stack(
        [patterns[:, :, :, query, key].mean(dim=0) for query, key in queries_keys], dim=-1
    )
    return HeadRoles(weights=weights)
