from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..core.metrics import measure
from ..data.tasks import CircuitTask
from ..model.adapter import require_circuits
from .circuits import (
    Baselines,
    Circuit,
    HeadId,
    baselines,
    behaviour,
    direct_logit_attribution,
    patch_heads,
)

"""
The field's question moved. It was "what is the circuit for this task"; it is
now "which way of finding one should anybody believe", and that turns the
technique itself into the thing under test.

So a technique is a registration here, the way a backend is a registration in
model/adapter.py and an experiment kind is one in experiment/runner.py. Each
one takes a task and hands back a score for every head, and the differences
between them are exactly the differences worth measuring:

- attribution reads the direct path off one decomposition. Exact, blind to
  everything a head does by changing what a later head reads, and cheap.
- patching asks the model, one forward pass per head. Causal, and the one that
  settles things, at a cost that is quadratic in the model.
- ablation asks the same question from the other side: not "does restoring
  this head bring the answer back" but "does removing it take the answer
  away". Those two come apart wherever the model has a second route, and a
  head that is sufficient but not necessary is the shape of that.
- eap is patching's first-order Taylor expansion -- one backward pass in place
  of one forward pass per head. It is the technique the recent literature runs
  at scale, and whether the approximation holds is a thing this module exists
  to measure rather than assume.
- random is the control. A circuit of the same size drawn without looking at
  the model, so that "our eight heads recover 90% of the span" can be read
  against what eight arbitrary heads recover.

Every ranking is by *absolute* score. A head that writes hard against the
answer is a head the task runs through, and the negative name movers are the
standing example: dropping them because their number is negative discards the
component the whole IOI story turns on.

A common pipe could be: build_task | rank | Ranking.select | verify
"""

class DiscoveryError(ValueError):
    """Raised when a technique is asked for something it cannot answer: no such method, no heads to select"""

@dataclass(frozen=True)
class Ranking:
    """Every head scored by one technique, with the units and the bill

    `units` is not decoration. attribution reports logits along the direct
    path, patching reports a fraction of the corruption's span, eap reports a
    first-order estimate of that same fraction and random reports nothing at
    all. Two of those are comparable as numbers and the rest are comparable
    only as orderings, which is why the comparison correlates ranks and
    intersects sets rather than subtracting scores.

    `layers` maps a row back to the layer it measured, for the same reason
    HeadEffects carries it: a sweep over part of the model still has a row 0.
    """
    scores: torch.Tensor
    method: str
    units: str
    layers: List[int]
    passes: int
    seconds: float
    baselines: Optional[Baselines] = None

    def heads(self) -> List[HeadId]:
        """Every head this ranking covers, in the row-major order `flat` uses"""
        return [(layer, head) for layer in self.layers for head in range(self.scores.shape[1])]

    def flat(self) -> List[float]:
        """The scores in a fixed order, so two techniques can be rank-correlated"""
        return [float(value) for value in self.scores.flatten()]

    def ranked(self, count: Optional[int] = None) -> List[Tuple[HeadId, float]]:
        """Heads ordered by how far they move the answer, largest absolute score first"""
        flat = self.scores.flatten()
        order = torch.argsort(flat.abs(), descending=True)
        width = self.scores.shape[1]
        chosen = order[: count if count is not None else len(order)]
        return [((self.layers[int(index) // width], int(index) % width), float(flat[index])) for index in chosen]

    def select(self, count: Optional[int] = None, frac: Optional[float] = None) -> Circuit:
        """Take the top heads as a circuit, by count or by share of the model

        `frac` is the top-K% selection the recent comparisons are reported at,
        and it is a share of every head the ranking covers rather than of the
        model, so a ranking swept over half the layers does not quietly
        double its K.

        The Circuit comes back with an empty `scores` list on purpose: that
        field is the greedy search's cumulative recovery curve, and a set
        chosen by threshold never walked one. Inventing a curve for it would
        make a top-K circuit indistinguishable from a grown one.

        Selection is by absolute score, so a set can come back whose
        restoration is *worse* than a random one -- take the top few heads of
        a late-layer sweep on IOI and most of them write against the answer.
        That is the ranking being right and faithfulness being the wrong
        question to ask of it: those heads are load-bearing, and what they
        bear is the deletion.
        """
        if (count is None) == (frac is None):
            raise DiscoveryError("select takes either count or frac, and needs exactly one of them")
        total = len(self.heads())
        if frac is not None:
            if not 0 < frac <= 1:
                raise DiscoveryError(
                    f"frac is a share of the {total} heads ranked, so it must be in (0, 1]; got {frac}"
                )
            count = max(1, round(frac * total))
        if not 1 <= count <= total:
            raise DiscoveryError(f"a circuit of {count} heads cannot be taken from the {total} this ranking covers")
        return Circuit(heads=[head for head, _ in self.ranked(count)], scores=[], threshold=0.0)

    def __str__(self) -> str:
        top = ", ".join(f"L{layer}H{head}" for (layer, head), _ in self.ranked(3))
        return f"{self.method}: {len(self.heads())} heads in {self.units}, {self.passes} passes, top {top}"

@dataclass(frozen=True)
class Technique:
    """One way of finding a circuit, and what it costs to believe it"""
    find: Callable[..., Ranking]
    units: str
    description: str
    cost: str

TECHNIQUES: Dict[str, Technique] = {}

def register_technique(units: str, description: str, cost: str) -> Callable:
    """Register a circuit-finding technique under a name, so a method can be named as data"""
    def decorate(find: Callable[..., Ranking]) -> Callable[..., Ranking]:
        TECHNIQUES[find.__name__.lstrip("_")] = Technique(
            find=find, units=units, description=description, cost=cost
        )
        return find
    return decorate

def technique_names() -> List[str]:
    """Every technique this module knows how to run, sorted"""
    return sorted(TECHNIQUES)

def rank(method: str, adapter, task: CircuitTask, layers: Optional[Sequence[int]] = None, **options) -> Ranking:
    """Score every head by the named technique"""
    if method not in TECHNIQUES:
        raise DiscoveryError(f"unknown technique '{method}'; known techniques are {technique_names()}")
    return TECHNIQUES[method].find(adapter, task, layers=layers, **options)

def _sweep(adapter, layers: Optional[Sequence[int]]) -> List[int]:
    """The layers a technique will cover, defaulting to all of them"""
    chosen = list(range(adapter.cfg.n_layers)) if layers is None else list(layers)
    outside = [index for index in chosen if not 0 <= index < adapter.cfg.n_layers]
    if outside:
        raise DiscoveryError(f"layers {outside} are outside the {adapter.cfg.n_layers} layers of '{adapter.cfg.id}'")
    if not chosen:
        raise DiscoveryError("a technique needs at least one layer to sweep")
    return chosen

# ------------------------------------------------------------- the techniques

@register_technique(
    units="logits",
    description="what each head wrote towards the answer along the direct path to the unembedding",
    cost="two forward passes, whatever the model's size",
)
def _attribution(adapter, task: CircuitTask, layers: Optional[Sequence[int]] = None, **options) -> Ranking:
    """Rank heads by direct logit attribution: exact, cheap, and blind to indirect paths"""
    adapter = require_circuits(adapter)
    chosen = _sweep(adapter, layers)
    with measure(items=1) as cost:
        result = direct_logit_attribution(adapter, task)
    rows = torch.tensor(chosen, dtype=torch.long)
    return Ranking(
        scores=result.heads.index_select(0, rows), method="attribution", units="logits",
        layers=chosen, passes=2, seconds=cost[0].seconds,
    )

@register_technique(
    units="recovery",
    description="how much of the corruption's span each head restores when written back on its own",
    cost="one forward pass per head, plus three for the baselines and the donors",
)
def _patching(adapter, task: CircuitTask, layers: Optional[Sequence[int]] = None, **options) -> Ranking:
    """Rank heads by activation patching: causal, expensive, and the reference the rest are read against"""
    adapter = require_circuits(adapter)
    chosen = _sweep(adapter, layers)
    with measure(items=1) as cost:
        effects = patch_heads(adapter, task, layers=chosen)
    return Ranking(
        scores=effects.effects, method="patching", units="recovery", layers=chosen,
        passes=len(chosen) * adapter.cfg.n_heads + 3, seconds=cost[0].seconds, baselines=effects.baselines,
    )

@register_technique(
    units="recovery",
    description="how much of the clean behaviour is lost when each head alone is knocked out",
    cost="one forward pass per head, plus three for the baselines and the donors",
)
def _ablation(adapter, task: CircuitTask, layers: Optional[Sequence[int]] = None, **options) -> Ranking:
    """Rank heads by what removing them costs, which is not the same question restoring them asks

    Patching asks whether a head is *sufficient* to bring the answer back;
    this asks whether it is *necessary* to keep it. Wherever the model has a
    second route to the answer both heads on it score low here and either can
    score high there, so the two rankings come apart exactly where redundancy
    lives -- which is the disagreement worth having a second technique for.
    """
    adapter = require_circuits(adapter)
    chosen = _sweep(adapter, layers)
    reference = baselines(adapter, task)
    donors = adapter.head_outputs(task.corrupted, layers=chosen)

    with measure(items=1) as cost:
        scores = torch.zeros(len(chosen), adapter.cfg.n_heads)
        for row, layer in enumerate(chosen):
            for head in range(adapter.cfg.n_heads):
                with adapter.patch(heads={layer: {head: donors[:, row, head]}}):
                    damaged = behaviour(adapter, task).logit_difference
                scores[row, head] = 1.0 - reference.recovery(damaged)
    return Ranking(
        scores=scores, method="ablation", units="recovery", layers=chosen,
        passes=len(chosen) * adapter.cfg.n_heads + 3, seconds=cost[0].seconds, baselines=reference,
    )

@register_technique(
    units="recovery (first order)",
    description="a gradient estimate of what patching each head would have recovered",
    cost="one backward pass and four forward passes, whatever the model's size",
)
def _eap(adapter, task: CircuitTask, layers: Optional[Sequence[int]] = None, **options) -> Ranking:
    """Rank heads by attribution patching: patching's first-order expansion, at constant cost

    The estimate is (clean - corrupted) . d(logit difference)/d(activation),
    with the gradient taken on the *corrupted* run, because the quantity being
    approximated is what happens when the corrupted run's activation is moved
    to the clean one's. Taking it on the clean run instead expands around the
    wrong point and quietly answers a different question.

    Summed over every position and every coordinate of the head, because
    patching restores the head at every position -- an estimate summed over
    less than the intervention it approximates is not an estimate of it.

    Divided by the span, so the number lands in the same recovery units
    patching reports and the two can be subtracted rather than merely ranked.
    Where they disagree, the linearization is what broke: a head whose effect
    is large is one the first-order term was never going to catch.
    """
    adapter = require_circuits(adapter)
    chosen = _sweep(adapter, layers)
    reference = baselines(adapter, task)
    with measure(items=1) as cost:
        clean = adapter.head_outputs(task.clean, layers=chosen)
        corrupted = adapter.head_outputs(task.corrupted, layers=chosen)
        gradients = adapter.head_gradients(task.corrupted, reference.io, reference.subject, layers=chosen)
        estimate = ((clean - corrupted) * gradients).sum(dim=(3, 4)).mean(dim=0)
    return Ranking(
        scores=estimate / reference.span, method="eap", units="recovery (first order)", layers=chosen,
        passes=5, seconds=cost[0].seconds, baselines=reference,
    )

IG_STEPS = 5

@register_technique(
    units="recovery (integrated)",
    description="eap with the gradient integrated along the path from corrupted to clean, not taken at a point",
    cost="IG_STEPS backward passes and four forward passes, whatever the model's size",
)
def _eap_ig(adapter, task: CircuitTask, layers: Optional[Sequence[int]] = None,
            steps: int = IG_STEPS, **options) -> Ranking:
    """Rank heads by attribution patching with integrated gradients (Hanna et al., 2403.17806)

    eap multiplies (clean - corrupted) by the gradient *at the corrupted
    point*, which is a first-order expansion and inherits the two failures of
    one: it overstates a component whose effect is not locally linear, and it
    returns nothing at all where the metric has saturated, because a flat
    region has a zero gradient no matter how much the component matters. A head
    behind a saturated softmax scores zero under eap and is not thereby
    unimportant.

    So the gradient is integrated along the straight line between the two
    inputs instead of read at one end of it. The estimate is
    (clean - corrupted) . mean over alpha of d(logit difference)/d(activation),
    with each gradient taken on a forward pass whose *input embeddings* are
    mixed (1 - alpha) * corrupted + alpha * clean. That is the `inputs` variant,
    which the MIB benchmark (Mueller et al., 2025) reports as the strongest of
    the family; the `activations` variant interpolates each node's activation
    instead and is a different estimator, not a detail of this one.

    Quadrature is the right Riemann sum over alpha = k/steps for k in 1..steps,
    the same choice the reference implementation makes. It is stated because it
    is a choice: the midpoint rule over the same budget is a different number,
    and one of the ways an IG result fails to reproduce is that nobody wrote
    down which one they took.

    This repo scores *nodes* -- Ranking.scores is [layer, head] -- so what is
    computed here is node attribution with integrated gradients, NAP-IG in the
    taxonomy that distinguishes them, using EAP-IG's mechanism at head
    granularity. The name keeps eap's for continuity with the technique it
    corrects; the units say `integrated` so a comparison cannot silently treat
    the two as the same estimator.

    Costs `steps` backward passes where eap costs one, which is the constant
    factor the paper trades for faithfulness. Everything else -- the corrupted
    baseline, the sum over positions and coordinates, the division by the span
    -- is eap's, so the two are subtractable and their difference is the part
    the linearization was getting wrong.
    """
    if steps < 1:
        raise ValueError(f"integrated gradients needs at least one step, got {steps}")
    adapter = require_circuits(adapter)
    chosen = _sweep(adapter, layers)
    reference = baselines(adapter, task)
    with measure(items=1) as cost:
        clean = adapter.head_outputs(task.clean, layers=chosen)
        corrupted = adapter.head_outputs(task.corrupted, layers=chosen)
        gradients = None
        for step in range(1, steps + 1):
            partial = adapter.head_gradients(
                task.corrupted, reference.io, reference.subject, layers=chosen,
                toward=task.clean, alpha=step / steps,
            )
            gradients = partial if gradients is None else gradients + partial
        estimate = ((clean - corrupted) * (gradients / steps)).sum(dim=(3, 4)).mean(dim=0)
    return Ranking(
        scores=estimate / reference.span, method="eap_ig", units="recovery (integrated)", layers=chosen,
        passes=4 + steps, seconds=cost[0].seconds, baselines=reference,
    )

MASK_STEPS = 40
MASK_RATE = 0.5
MASK_SPARSITY = 0.03

@register_technique(
    units="gate",
    description="a per-head gate learned by gradient descent under a sparsity penalty, not a score per head",
    cost="MASK_STEPS forward-and-backward passes, independent of how many heads there are",
)
def _mask(adapter, task: CircuitTask, layers: Optional[Sequence[int]] = None,
          steps: int = MASK_STEPS, rate: float = MASK_RATE, sparsity: float = MASK_SPARSITY,
          **options) -> Ranking:
    """Rank heads by a learned gate, the way Edge Pruning, DiscoGP and UGS do it

    Every other technique here scores heads one at a time and then hopes the
    scores add up -- patching restores one head per pass, eap estimates that
    restoration, and both leave the set-building to whoever reads the ranking.
    This one optimizes the set directly: each head gets a gate in [0, 1], its
    output becomes donor + gate * (clean - donor), and the gates are moved by
    gradient ascent on the logit difference minus a penalty on how many are
    open. What comes out is the answer to "which heads, together, are enough",
    which is a different question from "which head, alone, matters most" and is
    the question a circuit actually poses.

    The two come apart wherever the model has a second route. A pair of heads
    that back each other up scores low individually under patching -- removing
    either alone costs little -- and a mask cannot open both without paying
    twice, so it keeps one and the other's gate falls. That is the mask being
    right about the set and the solo ranking being right about the heads, and
    neither is a correction to the other.

    The donor is the corrupted activation, so a closed gate is exactly the
    counterfactual ablation patching performs -- MIB (Mueller et al., 2025)
    reports counterfactual ablation as what the strongest methods use, and it
    keeps a gate of zero here and a patch elsewhere the same intervention.

    What is optimized is *fidelity to the unablated model*, not the metric: the
    loss is the squared distance between the gated logit difference and the
    clean one, in units of the corruption span. Ascending on the metric instead
    was the first version of this and it is wrong in a way worth recording,
    because it looks right until you check which heads it keeps. IOI's strongest
    component at these layers is L10H7, a negative name mover with a patching
    score of -0.52 and a gate gradient of -1.25: it writes *against* the answer,
    so a mask maximizing the answer closes it, and closes the single head the
    circuit most depends on. A circuit is what the model uses, not what would
    help it, and only a loss that punishes moving *away* from the clean output
    in either direction can tell the two apart.

    Plain projected gradient ascent with gates clamped to [0, 1], rather than a
    sigmoid reparameterization or a hard-concrete relaxation. The relaxations
    buy a smoother path to a discrete mask and cost a temperature schedule that
    changes the answer; this is the version whose every number can be read off
    the loop. `sparsity` is the price of an open gate in logit-difference units
    and is the knob that decides how large a circuit comes back -- there is no
    setting of it that is neutral, so it is reported in the Ranking's options
    rather than buried.

    Scores are the gates themselves, so `units` is "gate" and not a recovery:
    a gate is not a fraction of the corruption span and must not be subtracted
    from one. Ranked by absolute value like everything else, which for gates in
    [0, 1] is just descending openness.
    """
    if steps < 1:
        raise ValueError(f"a mask needs at least one step, got {steps}")
    adapter = require_circuits(adapter)
    chosen = _sweep(adapter, layers)
    reference = baselines(adapter, task)
    with measure(items=1) as cost:
        donor = adapter.head_outputs(task.corrupted, layers=chosen)
        gates = torch.ones(len(chosen), adapter.cfg.n_heads)
        for _ in range(steps):
            value, gradient = adapter.head_gate_gradients(
                task.clean, donor, reference.io, reference.subject, gates, layers=chosen
            )
            # d(deviation^2)/d(gate), where deviation is how far the gated model has
            # drifted from the clean one in span units. Descend on that, and pay
            # `sparsity` for every gate still open.
            drift = (value - reference.clean) / reference.span
            gates = (gates - rate * (2 * drift * gradient / reference.span + sparsity)).clamp(0.0, 1.0)
    return Ranking(
        scores=gates, method="mask", units="gate", layers=chosen,
        passes=steps, seconds=cost[0].seconds, baselines=reference,
    )

@register_technique(
    units="none",
    description="a seeded shuffle, so a circuit's numbers can be read against a circuit of the same size",
    cost="no forward passes at all",
)
def _random(adapter, task: CircuitTask, layers: Optional[Sequence[int]] = None, seed: int = 0, **options) -> Ranking:
    """Score every head at random: the control every other technique has to beat

    Not a straw man. A model spreads a task across enough heads that an
    arbitrary handful of them recovers a real fraction of the span, and a
    technique whose circuit does not clear that is a technique that found the
    model rather than the task.
    """
    adapter = require_circuits(adapter)
    chosen = _sweep(adapter, layers)
    generator = torch.Generator().manual_seed(seed)
    return Ranking(
        scores=torch.rand(len(chosen), adapter.cfg.n_heads, generator=generator),
        method="random", units="none", layers=chosen, passes=0, seconds=0.0,
    )
