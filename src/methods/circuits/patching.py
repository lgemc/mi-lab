"""The causal half: write one activation back from the clean run and ask the model what changed.

Attribution infers from weights what a component contributed. Patching does
not infer anything -- it takes a corrupted run, restores one site to its clean
value, and reads the answer off the model. That is the measurement everything
else here is checked against, and it costs one forward pass per site, which is
why attribution is worth doing first.

The site is the thing to get right, and it is the same site three ways: `capture`
reads the residual stream leaving a block and the input to the attention output
projection, `patch` writes them, and `gradients` differentiates there. So
writing back what was already there is exactly a no-op, and every causal number
in this package is a difference against that no-op. A patch site that is not the
capture site breaks the property silently -- the numbers stay plausible.

Two grids and one restoration live here:

    patch_residual   where and when the model commits, as [layer, position]
    patch_heads      which heads carry the task, as [layer, head]
    restore          a whole set of heads written in at once, which is what
                     `search` grows and `verify` scores

Both grids carry `layers`, and that is not bookkeeping. A grid measured over a
subset still has row 0, and reporting that as layer 0 is how a partial sweep
becomes a confident statement about the embedding -- the same misreading
`Artifact._check_shapes` guards on the way out of the repo.

A common pipe could be: build_task | patch_heads | ranked | discover | verify
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ...core.config import Position
from ...core.readout import mean_score
from ...data.tasks import CircuitTask
from ...model.adapter import require_circuits
from ..common.components import HeadId
from ..common.errors import CircuitError
from ..common.intervention import head_patch
from ..common.span import Baselines, baselines


@dataclass
class PatchGrid:
    """How much of the clean behaviour each (layer, position) restores on its own

    `layers` says which layer each row is. A grid measured over a subset still
    has row 0, and reporting that as layer 0 is how a partial sweep turns into
    a confident statement about the embedding.
    """
    effects: torch.Tensor
    tokens: List[str]
    landmarks: Dict[str, int]
    baselines: Baselines
    layers: List[int] = field(default_factory=list)

    def best(self) -> Tuple[int, int, float]:
        """The single site that recovered the most, as (layer, position, recovery)"""
        index = int(self.effects.argmax())
        width = self.effects.shape[1]
        return self.layers[index // width], index % width, float(self.effects.flatten()[index])


def patch_residual(adapter, dataset: CircuitTask, layers: Optional[Sequence[int]] = None) -> PatchGrid:
    """Restore the clean residual stream at one (layer, position) at a time, into the corrupted run

    This is the map that says *where and when* the model commits: a bright cell
    means everything the answer needs has arrived at that layer and that
    position, and writing it back is enough on its own. Reading the bright
    cells in order of layer is reading the information move through the
    sentence.

    Every patch replaces the whole row at that layer with the corrupted run's
    own values, one position excepted. Overwriting a site with what it already
    held is exactly a no-op, so the difference between two cells is the
    position and nothing else.
    """
    adapter = require_circuits(adapter)
    if not len(dataset):
        raise CircuitError("an empty dataset has nothing to patch")
    reference = baselines(adapter, dataset)
    indices = list(layers) if layers is not None else list(range(adapter.cfg.n_layers))

    clean = adapter.capture(dataset.clean, layers=indices, position=Position.ALL)
    corrupted = adapter.capture(dataset.corrupted, layers=indices, position=Position.ALL)
    positions = clean.shape[2]

    effects = torch.zeros(len(indices), positions)
    for row, layer in enumerate(indices):
        for position in range(positions):
            donor = corrupted[:, row].clone()
            donor[:, position] = clean[:, row, position]
            with adapter.patch(residual={layer: donor}):
                patched = mean_score(adapter, dataset.corrupted, reference.readout)
            effects[row, position] = reference.recovery(patched)

    return PatchGrid(
        effects=effects,
        tokens=dataset.labels(adapter),
        landmarks=dataset.landmarks(adapter),
        baselines=reference,
        layers=indices,
    )


@dataclass
class HeadEffects:
    """How much of the clean behaviour each head restores on its own, as [layer, head]

    `layers` maps a row back to the layer it measured, so a sweep over part of
    the model still names its heads correctly.
    """
    effects: torch.Tensor
    baselines: Baselines
    layers: List[int] = field(default_factory=list)

    def ranked(self, count: Optional[int] = None) -> List[Tuple[HeadId, float]]:
        """Heads ordered by how much they move the answer, largest absolute effect first"""
        flat = self.effects.flatten()
        order = torch.argsort(flat.abs(), descending=True)
        width = self.effects.shape[1]
        chosen = order[: count if count is not None else len(order)]
        return [((self.layers[int(index) // width], int(index) % width), float(flat[index])) for index in chosen]


def patch_heads(adapter, dataset: CircuitTask, layers: Optional[Sequence[int]] = None) -> HeadEffects:
    """Restore one head's output at a time from the clean run into the corrupted one

    The causal answer to "which heads do this task". A head scoring near 1
    carries the whole task on its own; near 0 means the corruption never took
    anything from it; below 0 means its clean output pushes *against* the
    answer, which is a real and reproducible thing late heads do.

    The head is restored at every position, not only at the end, because a
    head that matters by moving something into the last position did its work
    earlier in the sentence.
    """
    adapter = require_circuits(adapter)
    reference = baselines(adapter, dataset)
    indices = list(layers) if layers is not None else list(range(adapter.cfg.n_layers))
    donors = adapter.head_outputs(dataset.clean, layers=indices)

    effects = torch.zeros(len(indices), adapter.cfg.n_heads)
    for row, layer in enumerate(indices):
        for head in range(adapter.cfg.n_heads):
            with adapter.patch(heads={layer: {head: donors[:, row, head]}}):
                patched = mean_score(adapter, dataset.corrupted, reference.readout)
            effects[row, head] = reference.recovery(patched)
    return HeadEffects(effects=effects, baselines=reference, layers=indices)


def restore(adapter, dataset: CircuitTask, reference: Baselines, heads: Sequence[HeadId], donors) -> float:
    """Recovery when this whole set of heads is written into the corrupted run at once

    donors is a full [batch, layer, head, seq, d_head] capture, so a layer
    index is a row index -- which is the reason these helpers never take a
    layer subset.
    """
    if not heads:
        return reference.recovery(reference.corrupted)
    with adapter.patch(heads=head_patch(heads, donors)):
        patched = mean_score(adapter, dataset.corrupted, reference.readout)
    return reference.recovery(patched)
