from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..core.config import Position
from ..core.metrics import logit_difference, recovery
from ..data.ioi import IOIDataset
from ..model.adapter import require_circuits

"""
A circuit study asks which parts of a model do a task, and it asks it twice.

Direct logit attribution is the correlational half: the residual stream is a
sum, so the logit difference the model ends on is the sum of what every head,
every MLP and the embedding wrote into it. That decomposition is exact and it
is cheap -- one forward pass answers for every component at once -- and it
only ever sees the *direct* path. A head that matters by changing what a later
head reads contributes nothing to it, and a head that writes the answer while
a later head deletes it looks like a hero.

Patching is the causal half and it is the one that settles things. Take a
corrupted run, write one activation back in from the clean run, and see how
much of the clean behaviour returns. Nothing is inferred from weights; the
model is asked. It costs one forward pass per site, which is why attribution
is worth doing first: it says where to look.

The two disagree, and the disagreement is the finding rather than a bug. In
GPT-2 small the late negative heads are the standard example -- large negative
attribution, and patching says the model needs them.

Everything here is measured against two baselines from the same prompts, so
every number is a fraction of the span the corruption opened. A result quoted
without that span is a number with no scale.

A common pipe could be: build_ioi | direct_logit_attribution | patch_heads | discover | verify
"""

class CircuitError(ValueError):
    """Raised when a circuit measurement is asked for something it cannot mean: no span, no heads, no positions"""

HeadId = Tuple[int, int]

ROLES = ("name mover", "s-inhibition", "duplicate token", "induction")

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

def _mean_difference(adapter, prompts: Sequence[str], io: Sequence[int], subject: Sequence[int]) -> float:
    """The mean logit difference over a batch of prompts"""
    return float(logit_difference(adapter.logits(list(prompts)), io, subject).mean())

def baselines(adapter, dataset: IOIDataset) -> Baselines:
    """Measure the clean and corrupted behaviour this dataset opens up

    A span near zero is fatal rather than merely disappointing: every later
    number divides by it, so a corruption that did not corrupt anything turns
    the whole study into noise amplified to look like a result.
    """
    adapter = require_circuits(adapter)
    io, subject = dataset.answers(adapter)
    clean = _mean_difference(adapter, dataset.clean, io, subject)
    corrupted = _mean_difference(adapter, dataset.corrupted, io, subject)
    found = Baselines(clean=clean, corrupted=corrupted, io=io, subject=subject)
    if abs(found.span) < 1e-6:
        raise CircuitError(
            f"the corruption moved the logit difference by {found.span:.2e}, so there is no span to recover; "
            "check that the corrupted prompts differ from the clean ones in the way you think"
        )
    return found

# ------------------------------------------------------------------ attribution

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

def direct_logit_attribution(adapter, dataset: IOIDataset, corrupted: bool = False) -> Attribution:
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
    io, subject = dataset.answers(adapter)
    decomposition = adapter.decompose(prompts)
    unembedding = decomposition.unembedding
    return Attribution(
        heads=unembedding.logit_difference(decomposition.heads, io, subject).mean(dim=0),
        mlps=unembedding.logit_difference(decomposition.mlps, io, subject).mean(dim=0),
        embedding=float(unembedding.logit_difference(decomposition.embedding, io, subject).mean()),
        offset=float(
            unembedding.logit_difference(decomposition.biases, io, subject).sum(dim=1).mean()
            + unembedding.offset(io, subject).mean()
        ),
        measured=_mean_difference(adapter, prompts, io, subject),
    )

# --------------------------------------------------------------------- patching

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

def patch_residual(adapter, dataset: IOIDataset, layers: Optional[Sequence[int]] = None) -> PatchGrid:
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
                patched = _mean_difference(adapter, dataset.corrupted, reference.io, reference.subject)
            effects[row, position] = reference.recovery(patched)

    return PatchGrid(
        effects=effects,
        tokens=dataset.token_labels(adapter),
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

def patch_heads(adapter, dataset: IOIDataset, layers: Optional[Sequence[int]] = None) -> HeadEffects:
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
                patched = _mean_difference(adapter, dataset.corrupted, reference.io, reference.subject)
            effects[row, head] = reference.recovery(patched)
    return HeadEffects(effects=effects, baselines=reference, layers=indices)

# --------------------------------------------------------------- head behaviour

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

# ------------------------------------------------------------ circuit and check

@dataclass
class Circuit:
    """A set of heads, and the record of how they were chosen"""
    heads: List[HeadId]
    scores: List[float] = field(default_factory=list)
    threshold: float = 0.0

    def __len__(self) -> int:
        return len(self.heads)

    def __str__(self) -> str:
        names = ", ".join(f"L{layer}H{head}" for layer, head in self.heads)
        return f"{len(self.heads)} heads: {names}" if self.heads else "empty circuit"

def _restore(adapter, dataset: IOIDataset, reference: Baselines, heads: Sequence[HeadId], donors) -> float:
    """Recovery when this whole set of heads is written into the corrupted run at once

    donors is a full [batch, layer, head, seq, d_head] capture, so a layer
    index is a row index -- which is the reason these helpers never take a
    layer subset.
    """
    if not heads:
        return reference.recovery(reference.corrupted)
    patch: Dict[int, Dict[int, torch.Tensor]] = {}
    for layer, head in heads:
        patch.setdefault(layer, {})[head] = donors[:, layer, head]
    with adapter.patch(heads=patch):
        patched = _mean_difference(adapter, dataset.corrupted, reference.io, reference.subject)
    return reference.recovery(patched)

def discover(
    adapter, dataset: IOIDataset, threshold: float = 0.8, max_heads: int = 12,
    effects: Optional[HeadEffects] = None,
) -> Circuit:
    """Grow a circuit greedily until restoring it alone reproduces the clean behaviour

    Candidates are tried in the order patch_heads ranked them, so the search is
    cheap rather than exhaustive, and it stops as soon as the set clears the
    threshold. The scores are kept per step because the shape of that curve is
    the result: a set whose recovery jumps at the third head and then flattens
    is a circuit, and one that climbs a little at every head is a model
    spreading the task across everything.

    Greedy means this finds *a* sufficient set, not the smallest one. verify's
    minimality column is what catches a passenger.
    """
    adapter = require_circuits(adapter)
    if max_heads < 1:
        raise CircuitError(f"a circuit needs room for at least one head, got max_heads={max_heads}")
    measured = effects or patch_heads(adapter, dataset)
    reference = measured.baselines
    donors = adapter.head_outputs(dataset.clean)

    chosen: List[HeadId] = []
    scores: List[float] = []
    for head_id, _ in measured.ranked(max_heads):
        chosen.append(head_id)
        scores.append(_restore(adapter, dataset, reference, chosen, donors))
        if scores[-1] >= threshold:
            break
    return Circuit(heads=chosen, scores=scores, threshold=threshold)

@dataclass
class CircuitReport:
    """What a circuit is worth: is it enough, is it needed, is any of it spare"""
    circuit: Circuit
    faithfulness: float
    necessity: float
    minimality: Dict[HeadId, float]
    baselines: Baselines

    def spare(self, tolerance: float = 0.05) -> List[HeadId]:
        """Heads whose removal costs the circuit almost nothing"""
        return [head for head, drop in self.minimality.items() if drop < tolerance]

    def __str__(self) -> str:
        return (
            f"{len(self.circuit)} heads  faithfulness {self.faithfulness:.2f}  "
            f"necessity {self.necessity:.2f}  spare {self.spare()}"
        )

def verify(adapter, dataset: IOIDataset, circuit: Circuit) -> CircuitReport:
    """Check a circuit three ways: sufficient, needed, and free of passengers

    - faithfulness: restore only these heads into the corrupted run. 1.0 means
      they are enough on their own.
    - necessity: do the opposite -- take a clean run and write *these* heads
      back to their corrupted values. 1.0 means the clean behaviour collapses
      without them, so they are not merely one of several routes to the answer.
    - minimality: per head, how much faithfulness drops when it alone is
      dropped. A head near zero is a passenger the greedy search picked up.

    This is weaker than the completeness test in Wang et al., which checks
    every subset rather than every single head, and it is weaker in the
    direction that matters: it can miss a pair of heads that are redundant
    together and load-bearing apart.
    """
    adapter = require_circuits(adapter)
    if not circuit.heads:
        raise CircuitError("an empty circuit has nothing to verify")
    reference = baselines(adapter, dataset)
    clean_donors = adapter.head_outputs(dataset.clean)
    corrupted_donors = adapter.head_outputs(dataset.corrupted)

    faithfulness = _restore(adapter, dataset, reference, circuit.heads, clean_donors)

    knocked: Dict[int, Dict[int, torch.Tensor]] = {}
    for layer, head in circuit.heads:
        knocked.setdefault(layer, {})[head] = corrupted_donors[:, layer, head]
    with adapter.patch(heads=knocked):
        broken = _mean_difference(adapter, dataset.clean, reference.io, reference.subject)

    minimality = {}
    for head_id in circuit.heads:
        rest = [other for other in circuit.heads if other != head_id]
        without = _restore(adapter, dataset, reference, rest, clean_donors)
        minimality[head_id] = faithfulness - without

    return CircuitReport(
        circuit=circuit,
        faithfulness=faithfulness,
        necessity=1.0 - reference.recovery(broken),
        minimality=minimality,
        baselines=reference,
    )
