"""A circuit you can run: joint weight and edge pruning, after DiscoGP (2407.03779).

Every circuit this repo has produced is a *selection* -- these heads, those
MLPs -- and a selection is not a runnable model. Deleting everything a
selection does not name deletes most of the network, and what comes back is
commas: phase 1b's extraction arm scored 0.0% retained for exactly that reason,
and DiscoGP names the pattern, reporting that circuit-based formulations
"typically fail when executed in isolation".

Its answer is to stop selecting and start optimizing. A gate is attached to
every weight, the weights themselves are frozen, and the gates are trained so
that the surviving skeleton still does the task. The paper reports 93-100% of
task performance at 1-7% of the weights, which no amount of choosing whole
components can reach, because the granularity of the choice is the limit.

Three terms, and the third is what makes it a circuit rather than a
compression:

    faith       the masked model still says what the model said (see
                `faith_kind`; this was a two-way logit comparison until
                2026-09-02, which the paper's term is not)
    sparsity    every open gate costs something
    complete    the *complement* is at chance -- with the mask reversed the
                model must be unable to do the task

Without `complete`, sparsity alone is satisfied by any subnetwork that happens
to work, including one that shares the mechanism with everything it dropped.
Requiring the complement to fail is what says the mask found where the
behaviour lives rather than merely somewhere it survives.

Gates are straight-through Gumbel-sigmoid: sampled and hard-thresholded going
forward, identity going back. The sampling is not decoration -- a deterministic
sigmoid gate lets the optimizer sit forever at 0.5, which is neither open nor
closed and scores well because half of everything is still there.

Weights are masked through `torch.func.functional_call` rather than copied into
the parameters in place. The reference implementation assigns masked values
onto frozen parameters with `copy_`; that is faster and it is also how a
gradient quietly stops flowing to the thing being trained, so this takes the
functional route and pays for it in memory.

A common pipe could be: task | gates | faith + sparsity + complete | sheaf

Not implemented here: edge gates. `edge_patch` in the backend is the substrate
for them and the two compose, but this module prunes weights only, which is the
half that the node-level `mask` technique in circuits/techniques.py cannot
express.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as functional

from ...data.tasks import CircuitTask
from ...model.adapter import require_circuits
from ...telemetry.journal import Journal
from ..common.errors import SheafError
from .faith import faith_for
from .forward import (
    attribution,
    chunk_rows,
    faith_term,
    first_token_accuracy,
    logit_pairs,
    ranking_accuracy,
    split_rows,
)
from .gate import pin, protected, schedule, target_schedule
from .gateable import gateable, span
from .units import Units, init_from


@dataclass
class Sheaf:
    """A pruned skeleton and what it costs to run it

    `density` is the fraction of gates left open, and it is the number the
    performance has to be read against: 90% of the task at 90% of the weights
    is not a finding.
    """
    gates: Dict[str, torch.Tensor]
    density: float
    n_parameters: int
    n_open: int
    accuracy: float
    train_accuracy: float
    complement_accuracy: float
    baseline_accuracy: float
    history: List[dict]
    layers: Optional[List[int]] = None
    # Reported beside the weight density, never folded into it. The paper's
    # "1%-7%" is of weights *and connections*, and one number covering both
    # hides which half did the pruning.
    n_edges: int = 0
    n_edges_open: int = 0
    edge_density: Optional[float] = None
    # The open edges themselves. `n_edges_open` is a count, and a count is not
    # a circuit: an edge run's whole product is *which* paths stayed, and
    # nothing here recorded them, so every edge run so far was unreadable.
    edges: Optional[List[tuple]] = None
    # Held-out argmax over the whole vocabulary, masked and full. `accuracy` is
    # a two-way comparison between the answer and one distractor, so its floor
    # is chance and a mask can hold it while the distribution around those two
    # logits collapses -- which is what ' mind mind mind' is. Cheap (one
    # forward pass, no generation) and it is the number the ranking hides.
    first_token: Optional[float] = None
    baseline_first_token: Optional[float] = None
    # The best mask the run ever held, and what it scored -- separately from
    # the last one, which is what `gates` is. A pruning run walks down a
    # density/faithfulness curve and stops when the step budget runs out, not
    # when it is doing well: qwen3-1.7b ioi held 1.000 at 12.6% density at step
    # 1975 and finished at step 1999 with 16.6% and 0.573, and the good mask
    # was never written anywhere. Chosen on the in-run probe and then rescored
    # here on the whole held-out set, so this number and `accuracy` mean the
    # same thing.
    best_gates: Optional[Dict[str, torch.Tensor]] = None
    # The open edges at that step. Same reason and the same bug: an edge run's
    # product is its edge list, and saving the last one saves whatever the
    # schedule was holding when the step budget ran out.
    best_edges: Optional[List[tuple]] = None
    best_accuracy: Optional[float] = None
    best_density: Optional[float] = None
    best_step: Optional[int] = None
    # What the metric scores when the mask knows nothing. Measured, not assumed
    # to be 0.5: `load_bearing`'s `shut` is this same quantity and lands at
    # 0.477 on the 1.7B's translation frame, so a circuit at 0.72 has recovered
    # 46% of what there was to recover and not 72% of anything.
    chance: float = 0.5

    @property
    def recovered(self) -> Optional[float]:
        """The share of the range between chance and the full model that the mask holds"""
        span_ = self.baseline_accuracy - self.chance
        return None if span_ <= 0 else (self.accuracy - self.chance) / span_
    # Gates held open outside the search (`protect`); inside `n_open`.
    n_pinned: int = 0
    # Steps on which the learned price was reset because the constraint held.
    n_restarts: int = 0
    # The head/neuron/block gates of a `granular` run: the report for the
    # artifact, and the logits for the `-units.pt` file beside the mask.
    units: Optional[dict] = None
    unit_logits: Optional[Dict[str, torch.Tensor]] = None

    def __str__(self) -> str:
        # The band is in the string because `density` is a fraction of what was
        # gated, not of the model: 1% open across seven layers and 1% open
        # across twenty-eight are different claims wearing the same number.
        band = "all layers" if self.layers is None else f"layers {span(self.layers)}"
        edges = ("" if self.edge_density is None
                 else f" · {self.edge_density:.2%} of {self.n_edges} edges")
        if self.units is not None:
            counts = self.units["counts"]
            edges += " · " + ", ".join(f"{entry['open']}/{entry['total']} {family}s"
                                       for family, entry in counts.items())
        # The recovered fraction, not the raw ranking. `accuracy` runs from
        # chance to the full model's, so 0.72 against a 0.48 floor is not 72%
        # of anything -- it is 46% of the range the mask could have recovered,
        # and quoting the raw number has read as twice the result it is.
        span_ = self.baseline_accuracy - self.chance
        recovered = ("" if span_ <= 0
                     else f" [{(self.accuracy - self.chance) / span_:.0%} of range]")
        first = ("" if self.first_token is None else
                 f" · first token {self.first_token:.3f}"
                 + ("" if self.baseline_first_token is None
                    else f" of {self.baseline_first_token:.3f}"))
        # The best mask is named whenever it is not the last one, because a run
        # that ended worse than it was is a run whose headline number is an
        # accident of where the step budget ran out.
        peak = ""
        if self.best_accuracy is not None and self.best_step is not None:
            peak = (f" · BEST {self.best_accuracy:.3f} at {self.best_density:.2%} "
                    f"(step {self.best_step})")
        return (f"sheaf: {self.density:.2%} of {band} open{edges} · "
                f"held-out {self.accuracy:.3f}{recovered} "
                f"(train {self.train_accuracy:.3f}) against {self.baseline_accuracy:.3f} full · "
                f"complement {self.complement_accuracy:.3f}{first}{peak}")

def load_bearing(adapter, task: CircuitTask, layers: Optional[Sequence[int]] = None,
                 rows: Optional[Sequence[int]] = None) -> Dict[str, float]:
    """Score the task with the band fully open and with every gate in it shut

    A band that can be deleted outright without moving the metric cannot host a
    circuit for it, and pruning inside one is free: sparsity closes every gate,
    faithfulness never objects because the layers *outside* the band still
    answer, and the run reports a density near zero at full accuracy. That is
    the shape of a result with none of the content, and it is the exact shape a
    reader would quote.

    Measured before the run rather than inferred after it. The first band tried
    here -- layers 21, 23, 24 and 26 of a 28-layer model, chosen because phase
    1b's candidate lives in them -- scored 1.000 open and 1.000 shut on the
    word-level translation task, so the 0.0008% density it went on to report
    was those four layers being irrelevant rather than a circuit being found.
    """
    targets = gateable(adapter, layers)
    originals = {name: parameter.detach().clone() for name, parameter in targets.items()}
    chosen = list(range(len(list(task.clean)))) if rows is None else list(rows)
    with torch.no_grad():
        shut = {name: torch.full(parameter.shape, -1.0, dtype=torch.float32,
                                 device=parameter.device)
                for name, parameter in targets.items()}
        return {
            "open": ranking_accuracy(logit_pairs(adapter, task, chosen, None, originals, 1.0,
                                     deterministic=True)),
            "shut": ranking_accuracy(logit_pairs(adapter, task, chosen, shut, originals, 1.0,
                                     deterministic=True)),
        }

def prune(adapter, task: CircuitTask, steps: int = 500, rate: float = 0.1,
          sparsity: float = 1.0, completeness: float = 0.3, temperature: float = 1.0,
          init: float = 1.0, batch: int = 64, max_times: float = 1000.0,
          warmup: Optional[int] = None, holdout: float = 0.25,
          layers: Optional[Sequence[int]] = None,
          journal: Optional[Journal] = None, probe_every: int = 10, probe_size: int = 0,
          seed: Optional[int] = None, edge_sparsity: float = 0.0,
          faith_kind: str = "pair", anneal: bool = False,
          target: Optional[float] = None, protect: float = 0.0,
          dual_rate: Optional[float] = None, dual_restart: bool = False,
          granular: Optional[Sequence[str]] = None, attribute: int = 0,
          init_low: float = 2.0, gate_weights: bool = True) -> Sheaf:
    """Learn a weight mask that does the task and whose complement cannot

    `granular` names unit families -- `head`, `kv`, `neuron`, `block` --
    and adds a gate per unit over the gate per weight, multiplied in
    (units.py; Haider et al., COLM 2026): a unit the task can spare closes
    by one parameter, and the density is still the fraction of weights
    open under every gate above them. Blocks are too coarse for the price
    to track and are not a default (see `Units.build`). The
    returned `gates` are then the effective boolean mask, since no single
    logit says whether a weight is open. `attribute` warm-starts every gate
    from first-order attribution on `attribute` batches of the full model
    (`attribution`): logits are the percentile rank of |w * dL/dw| mapped
    onto [`init_low`, `init`] -- the least implicated weight in each tensor
    starts `init_low` above the threshold and the most implicated at `init`.

    `init` starts every gate open -- a logit of 3 is a sigmoid of 0.95 -- so the
    search prunes a working model down rather than growing one from nothing.
    Starting closed makes the faithfulness term flat: a model with no weights
    has no gradient pointing at which weight to restore first. How open matters
    by model: the reference's 1.0 is a sigmoid of 0.73, and the *sampled*
    network at step 0 has 27% of its weights dropped at random. GPT-2 small
    tolerates that; the 1.7B does not (faith 11 nats at step 0 on translation,
    journals/20260902-214006), and every gradient after is taken on a model
    that is already broken.

    `temperature` is the backward sharpness and nothing else (see
    `gumbel_sigmoid`); the reference trains weight masks at 0.01.

    `anneal` shrinks the gate noise linearly from 1 to 0 across the run, so the
    mask being trained converges on the mask `logits > 0` that is evaluated and
    saved. Off, the two are the same network only where the logits have left
    the boundary, and the 1.7B's did not: 60% of them finished within 0.5 of
    zero (results/qwen3-1.7b-sweep/nll-s01), where a sample and the threshold
    disagree on nearly half the gates.

    `sparsity` and `completeness` are the two prices, and neither has a neutral
    setting. Their ratio decides whether the answer is a small circuit that
    barely works or a large one that certainly does, which is a choice about
    what is being claimed and is returned in `history` rather than hidden in
    the loop.

    `sparsity` is the *starting* price and `max_times` the factor it ramps to.
    The defaults are the reference implementation's -- 1.0 rising to 1000.0
    across the run -- and they matter more than they look: a first version of
    this held the price constant at 20 and pruned 11% of the weights where the
    paper reports 93-99%.

    `target` replaces that price with a density and learns the price. The
    sparsity term becomes the Lagrangian of Wang, Wohlwend & Lei (1910.04732)
    and CoFi (2204.00408), `l1 * (open - t) + l2 * (open - t)^2`, with `l1`
    and `l2` climbing by gradient ascent on the same step the gates descend:
    while the mask is denser than `t` the price rises until the gates give
    way, and once it is sparser the price falls and faith wins weights back.
    The price stops being a number tuned on one model and carried to another
    (0.1 x 1000 is GPT-2 IOI's, and it is what the 1.7B runs inherited), and
    Gallego-Posada et al. (NeurIPS 2022) show the penalty form is the less
    stable of the two. On GPT-2 IOI a 3% target reached 2.22% at held-out
    0.969 against the ramp's 2.66% / 0.945 (results/gpt2-sweep/nll-t003). It
    was written to rescue the 1.7B, and did not: that collapse was `nll`'s
    target token, below. `t` follows `target_schedule`, and in this mode
    `warmup` defaults to half the run so the mask trains at its target for
    the other half. `sparsity` and `max_times` are ignored. The weights only:
    an edge price, if any, is still `edge_sparsity` on the ramp.

    How the multipliers climb is `dual_rate` and `dual_restart`, both from
    Gallego-Posada et al. (NeurIPS 2022, 2208.04425). By default they sit in
    the same AdamW as the gates, and Adam normalizes: the multiplier moves
    about `rate` per step whatever the gap's size or sign, so a price built
    over 1500 steps of standing gap takes as long to unwind. On the 1.7B at a
    20% target the density crossed the target at step 1650 and kept falling
    to 12.6% while `l1` came down from 148 to 134 (results/qwen3-1.7b-sweep/
    kl-t02): the mask finished 40% sparser than asked and paid for it. With
    `dual_rate` the multipliers take plain gradient ascent at that rate --
    `l1 += dual_rate * gap`, proportional to the gap, projected to stay
    non-negative since the constraint is `density <= t` -- and `dual_restart`
    resets both to zero on any step the constraint holds: the price is
    rebuilt from nothing if the mask grows back past the target, and a mask
    that has reached its target trains on faith alone until it does.

    `protect` pins the top fraction of weights by magnitude open, outside the
    search. The faithfulness gradient on a gate is the first-order cost of
    closing it, and on Qwen3-1.7B that estimate is 25x short for the largest
    weights: closing the top 0.01% by |w| is priced at 0.36 nats and costs 8.2,
    and their gates draw ten times the median gradient with a fifth of them
    still below it (scratchpad gategrad, 2026-09-03). Pinned gates count as
    open in the density, because they are: the circuit runs on them. Written
    against the 1.7B collapse and not its cause -- with 0.1% pinned the mask
    still went to chance at 94% (results/qwen3-1.7b-sweep/diag-protect), and
    the collapse was the `nll` target token, below. Whether a run needs it is
    open; the fragility it answers is real.

    `faith_kind` "nll" is the likelihood of the token the *full model*
    predicts, and that token has to be the answer for the term to mean what
    the paper means. On GPT-2 IOI it is the name. On the translation frame the
    1.7B's argmax is ` "` for 113 of 128 prompts -- it says ` "cost" and` --
    so nll trained the mask to open a quote, the mask learned it (faith at
    0.001 while the ranking probe sat at chance), and three runs at three
    prices, two inits, with and without annealing all collapsed between 95%
    and 98% density, the cost of finding "always say quote". The circuits
    generate ` " " " "`: the objective's optimum. Where the answer is not the
    argmax, train "kl", which keeps the answer's rank inside the distribution.

    The constraint is measured on the *hard* density, `logits > 0`, the
    number every artifact reports; the gates descend through the relaxed
    `open`, the only one with a gradient. CoFi constrains the expected L0,
    and doing that here landed a 3% target at 2.0% hard with `open` still
    at 6.5% and the price still climbing (results/gpt2-sweep/nll-t003 v1):
    closed gates just under zero count as open in the relaxed cost and never
    in the mask. It also cost the first quarter of that run: `open` starts
    at sigmoid(init), below a target ramp that starts at 1.0, so the price
    went to -18 and spent 450 steps forcing gates open against faith. The
    hard density starts at exactly 1.0.

    `faith_kind` chooses what faithfulness means, and only one of the three is
    the paper's.

    "nll" is DiscoGP's own term, `-sum_i log p_m(y-hat_i | x_i)`: the likelihood
    the masked model assigns the token the *full* model predicted, over the
    whole vocabulary.

    "pair" is what this module did first and it is not the paper's, whatever
    the commit that added it claimed. Cross-entropy over just the good and bad
    logits is a two-way choice, and a mask satisfies it by depressing one logit
    relative to the other while the rest of the distribution collapses. The
    circuit it produced on IOI ranked 0.938 on unseen prompts and generated
    " Mary Emma Rose the Rose" -- 0.055 where the model scores 0.953. Kept
    because every result before 2026-09-02 was measured with it.

    "kl" is the soft-target relative of "nll", matching the full distribution
    rather than its argmax. The paper uses KL to evaluate rather than to train.
    Switching from "pair" to "kl" took first-token generation from 0.055 to
    0.625, which is the size of the error "pair" was hiding.

    "gold" is "nll" with the task's own answer as the label instead of the
    full model's argmax: `-sum_i log p_m(y_i | x_i)` over the whole
    vocabulary. Not the paper's term, and not faithfulness to the model in
    the paper's sense -- a circuit trained on it may be right where the full
    model is wrong. It exists because "kl" trains the mask to reproduce
    everything the full model does at that position, and on the 1.7B's
    translation frame that is the wrong word on 29% of prompts and a
    ` \nSpanish:` continuation on all of them; the ranking and the
    first-token probes score the *answer*. Two masks trained on "kl" at 12.6%
    and 19.1% (results/qwen3-1.7b-sweep/kl-t02, kl-t02-dual) ranked 0.89 and
    0.85 and got the first token on a third of prompts, and the denser one
    was the worse, which is the signal and not the density.

    `gate_weights` False leaves every weight open and prunes *only* edges,
    which is not DiscoGP but is the experiment DiscoGP's own results make
    hard to run at this scale. A gate per weight on the 1.7B is 1.4e9 free
    bits trained against ~350 distinct prompts, and the strong lottery-ticket
    results (Ramanujan et al. 2020; Malach et al. 2020) say a mask with that
    much freedom is not finding a subnetwork, it is training one -- which is
    what train 0.99 against held-out 0.72 looks like from the inside, on
    every run of the sweep at every density. Dropping to ~14k edge gates
    cuts the search space by five orders of magnitude and costs no
    expressiveness the *circuit* claim needs: an edge is the unit the claim
    is about. It also removes the 23 GiB of AdamW state, so a run is minutes
    rather than two hours. With a `target`, the learned price rides the edges,
    because they are then the only gates there are.

    `edge_sparsity` turns on the other half of the method. DiscoGP prunes
    "not only subsets of edges in an LM's computation graph but also the
    model's weight parameters", and weights alone is what this was until now --
    which is the standing explanation for the result weights alone produced: a
    mask scoring 0.938 on a two-way ranking test while generating ` Mary Emma
    Rose the Rose`, having deleted the model's name-mover heads at no cost. A
    weight mask may distort the output distribution freely so long as two
    logits keep their order; an edge mask constrains which paths carry signal
    at all.

    It is priced separately, and that is not tuning. There are ~2k edges
    against 85M weights on GPT-2 small, so one shared coefficient makes the
    edge term four orders of magnitude smaller than the weight term and the
    edge gates never move. Zero disables edges entirely and the function is
    exactly what it was.

    `seed` seeds torch's global generator, which is the only thing that makes
    a run repeatable: the gates are sampled by `gumbel_sigmoid` on every forward
    pass, from the global RNG, so two runs of identical arguments otherwise
    produce different masks. The task's own `seed` never reached this -- it
    selects which prompts are drawn and nothing about the sampling -- so a
    "seed sweep" varying it would have measured the wrong source of variance.

    `journal` streams every step to disk as it happens. Without one this
    function is silent for its whole duration and returns everything at the
    end, which on the whole 1.7B model is two hours of blank terminal and
    nothing at all if the process is killed -- and it has been, by the driver,
    at the two-second mark and by a stale split at the ninety-minute mark.
    `probe_size` is how many held-out rows that probe scores, defaulting to
    `batch` for the runs taken before it existed. A curve is only as resolvable
    as its denominator: eight rows is a standard error of 0.18, which is most of
    the range between chance and the full model on a two-way task.

    `probe_every` throttles the two metrics that are not free: `density`
    reduces over every gate, and `hard_accuracy` is a forward pass of the
    *thresholded* mask on held-out rows -- the mask the run is quoted on,
    which the loss terms never touch because they are measured on sampled
    ones. That gap is where a run collapses without the curve saying so.
    """
    adapter = require_circuits(adapter)
    if steps < 1:
        raise SheafError(f"pruning needs at least one step, got {steps}")
    if seed is not None:
        # Global rather than a threaded Generator: gumbel_sigmoid is called once
        # per gated parameter per pass and a generator argument would have to
        # reach every one of them. A process runs one prune, so the global state
        # is not shared with anything that would notice.
        torch.manual_seed(seed)
    targets = gateable(adapter, layers)
    if not targets:
        raise SheafError("no maskable weights found; every block was norms and embeddings")

    originals = {name: parameter.detach().clone() for name, parameter in targets.items()}
    for parameter in targets.values():
        parameter.requires_grad_(False)
    # float32 whatever the weights are. `full_like` inherited the model's dtype,
    # which is bfloat16 on every model here larger than GPT-2, and bfloat16
    # carries eight mantissa bits: AdamW's second moment underflows, and
    # `open_cost` below sums sigmoid over every gate, where accumulating 1.4e9
    # terms in bfloat16 stops adding once the running total passes 256.
    device = next(iter(targets.values())).device
    gates = ({name: torch.full(parameter.shape, init, dtype=torch.float32,
                               device=parameter.device).requires_grad_(True)
              for name, parameter in targets.items()} if gate_weights else {})
    # Of `targets`, not of `gates`: with the weights ungated there are no gate
    # tensors to count and the density is still a fraction of the same band.
    total = sum(parameter.numel() for parameter in targets.values())
    if not gate_weights:
        if edge_sparsity <= 0.0 and target is None:
            raise SheafError(
                "gate_weights False prunes edges only, so something has to price them: "
                "give --edge-sparsity, or --target for a learned price on the edge density"
            )
        if protect > 0.0 or granular or attribute > 0:
            raise SheafError(
                "protect, granular and attribute all shape the *weight* gates, and "
                "gate_weights False has none"
            )
    pinned = protected(originals, protect) if gate_weights else {}
    units = None
    if granular:
        units = Units.build(adapter, {name: tuple(g.shape) for name, g in gates.items()}, init,
                            device, families=tuple(granular))
    pin(gates, pinned)
    # One scalar per (source, destination) the residual stream admits. Cheap
    # next to the weights -- 2028 against 85M on GPT-2 small -- and the half
    # that says which paths exist rather than how strong they are. Built before
    # the optimizer rather than added to it after: an edges-only run has no
    # weight gates, and AdamW refuses to be constructed on an empty list.
    edge_ids = list(adapter.edges()) if (edge_sparsity > 0 or not gate_weights) else []
    edge_logits = None
    if edge_ids:
        edge_logits = torch.full((len(edge_ids),), init, dtype=torch.float32,
                                 device=device).requires_grad_(True)
    optimizer = torch.optim.AdamW(
        list(gates.values()) + (units.parameters() if units else [])
        + ([edge_logits] if edge_logits is not None else []), lr=rate)

    def hard_open(name: str, logits: torch.Tensor) -> torch.Tensor:
        opened = logits > 0
        return opened & units.hard(name, logits.ndim) if units is not None else opened

    def hard_count() -> int:
        if not gate_weights:
            return total
        return int(sum(int(hard_open(name, g).sum()) for name, g in gates.items()))

    # a mask trained and scored on the same prompts memorizes them. The first run
    # of this reported 0.02% of weights open at accuracy 1.000, which is not a
    # circuit, it is eight examples learned by 20k weights. DiscoGP splits; so does
    # this, and `accuracy` below is the held-out number.
    train_rows, test_rows = split_rows(list(task.clean), holdout)
    with torch.no_grad():
        baseline = ranking_accuracy(logit_pairs(adapter, task, test_rows, None, originals, temperature,
                                    deterministic=True))
    # A fixed set of held-out rows for the probe below, the same rows every time
    # so the curve is one quantity over the run rather than one per draw.
    #
    # `probe_size` because `batch` was the wrong denominator: at batch 8 the
    # curve is eight examples, +-0.18, and it read 1.000 -> 0.375 -> 0.875 on
    # consecutive probes of a run that was descending smoothly. That is fine for
    # what this was written as -- a collapse detector, which only has to tell
    # zero from not-zero -- and useless as the accuracy curve anyone actually
    # wants to plot. It costs one forward pass per probe either way; the rows in
    # it are free. 0 keeps the old behaviour.
    wanted = probe_size if probe_size > 0 else batch
    probe_rows = test_rows[:max(1, min(len(test_rows), wanted))]

    faith_score = faith_for(faith_kind)
    # The reference distribution, taken once from the unmasked model. The
    # weights are frozen, so what the full model says never changes and
    # recomputing it every step would only pay for the same numbers again.
    reference = None
    if faith_kind == "gold":
        # The label is the task's, and the full model is never consulted.
        io = task.readout(adapter).positive
        reference = {row: torch.tensor([io[row]], device=device) for row in train_rows}
    if faith_kind in ("kl", "nll"):
        with torch.no_grad():
            reference = {}
            for row in train_rows:
                full = logit_pairs(adapter, task, [row], None, originals, temperature,
                              deterministic=True, whole=True)
                # KL wants the distribution; NLL wants the token the full model
                # would actually emit, which is what the paper's y-hat is.
                reference[row] = (full.log_softmax(dim=-1) if faith_kind == "kl"
                                  else full.argmax(dim=-1))

    if attribute > 0:
        if not init_low < init:
            raise SheafError(f"init_low must be below init, got {init_low} and {init}")
        scores = attribution(adapter, task, train_rows, originals, faith_score, reference,
                             temperature, batch, attribute)
        init_from(scores, gates, units, init_low, init)
        del scores
        pin(gates, pinned)

    if target is not None and not 0.0 < target <= 1.0:
        raise SheafError(f"a density target is a fraction in (0, 1], got {target}")
    if warmup is None:
        warmup = steps // 2 if target is not None else steps
    # The learned price. Two scalars, ascended rather than descended: their
    # gradient is negated before the step below, so one AdamW on both sides
    # of the saddle. Started at zero, so step 0 is faith alone.
    multipliers = None
    if target is not None:
        multipliers = torch.zeros(2, dtype=torch.float32, device=device)
        if dual_rate is None:
            multipliers.requires_grad_(True)
            optimizer.add_param_group({"params": [multipliers], "weight_decay": 0.0})
    elif dual_rate is not None or dual_restart:
        raise SheafError("dual_rate and dual_restart shape the learned price: give a target")
    restarts = 0
    history = []
    # The best mask seen, by the probe. `gates` is the *last* mask and that is
    # not the same thing; see Sheaf.best_gates.
    best: Dict[str, Any] = {"accuracy": None, "density": None, "step": None,
                            "gates": None, "edges": None}
    for step in range(steps):
        price = schedule(step, sparsity, max_times, warmup)
        goal = target_schedule(step, target, warmup) if target is not None else None
        noise = 1.0 - step / max(1, steps - 1) if anneal else 1.0
        chunk = chunk_rows(train_rows, step, batch)
        optimizer.zero_grad()
        # The two masked passes are backed through one at a time, and the
        # gradients accumulate into the same `.grad` -- which is the identical
        # quantity `(faith + price * open_cost + completeness * complete)
        # .backward()` produces, because a gradient of a sum is a sum of
        # gradients. What it does not do is hold both graphs at once. Every
        # tensor gumbel keeps for the backward pass is one float32 per *weight*,
        # not per example, so two live graphs is the whole memory profile of
        # this loop: on a 352M-gate band the single-backward version asked the
        # driver for more than a 27 GiB pool had and was killed two seconds in.
        # This is also why shrinking `batch` barely helps and narrowing the
        # layer band helps a great deal.
        pairs = logit_pairs(adapter, task, chunk, gates, originals, temperature,
                       edge_logits=edge_logits, edge_ids=edge_ids,
                       whole=faith_score.whole, noise=noise, units=units)
        faith = faith_term(pairs, chunk, faith_score, reference)
        # every open gate costs something, measured on the relaxed probability so
        # the term has a gradient where the hard gate does not
        if not gate_weights:
            # No weight gates, so nothing to price and nothing to differentiate;
            # `sum(())` would be a plain 0 and `float(...detach())` would fail.
            open_cost = torch.zeros((), device=pairs.device)
        elif units is None:
            open_cost = sum(torch.sigmoid(g).sum() for g in gates.values()) / total
        else:
            # a weight's expected openness is the product of every gate above it
            open_cost = sum((torch.sigmoid(g) * units.relaxed(name, g.ndim)).sum()
                            for name, g in gates.items()) / total
        edge_cost = (torch.sigmoid(edge_logits).mean() if edge_ids
                     else torch.zeros((), device=pairs.device))
        # Ramped on the same schedule as the weights, not held constant. A
        # constant edge price reproduced exactly the failure `schedule` was
        # written to prevent: at 1.0 against a weight price starting at 0.01 it
        # outweighed the weights a hundredfold at step 0, closed 62% of the
        # edges within 333 steps, and `faith` never came below the tie point --
        # train accuracy finished at 0.509, chance, having never fit at all.
        edge_price = schedule(step, edge_sparsity, max_times, warmup) if edge_ids else 0.0
        hard_density = None
        if multipliers is not None:
            # A reduce over every gate on every step, which the probe below
            # throttles; here it is the constraint, and it is one comparison
            # and one sum per tensor against a forward pass of the model.
            with torch.no_grad():
                hard_density = (float((edge_logits > 0).float().mean()) if not gate_weights
                                else hard_count() / total)
            gap = hard_density - goal
            # The marginal price of an open gate under the Lagrangian,
            # `l1 + 2 l2 (density - t)`, applied to the relaxed cost so the
            # gates have a gradient. Detached: the multipliers are ascended
            # on the gap itself, below, not through this product.
            price = float((multipliers[0] + 2.0 * multipliers[1] * gap).detach())
        # With the weights ungated the edges are the only gates, so the learned
        # price is theirs: `price` is what the Lagrangian moves against the
        # density gap, and `edge_price` is the fixed ramp for the joint runs.
        if not gate_weights and multipliers is not None:
            edge_price = price
        sparse_term = price * open_cost
        (faith + sparse_term + edge_price * edge_cost).backward()
        restarted = False
        if multipliers is not None:
            ascent = torch.tensor([gap, gap * gap], dtype=torch.float32, device=multipliers.device)
            with torch.no_grad():
                if dual_restart and gap <= 0.0:
                    # The constraint holds: the price is a debt from steps
                    # when it did not, and there is nothing left to pay for.
                    multipliers.zero_()
                    restarted = True
                    restarts += 1
                elif dual_rate is not None:
                    # Gradient ascent on `l1 gap + l2 gap^2` at its own rate,
                    # so a small gap moves the price a little and a large one
                    # a lot; clamped because a negative price would be paying
                    # the mask to grow past an upper bound.
                    multipliers.add_(dual_rate * ascent).clamp_(min=0.0)
            if dual_rate is None:
                # The same ascent through the gates' AdamW, as the negated
                # gradient. On a restart no gradient at all, so AdamW skips
                # the tensor rather than stepping the zero it was just set to.
                multipliers.grad = None if restarted else -ascent
        faith_value, open_value = float(faith.detach()), float(open_cost.detach())
        edge_value = float(edge_cost.detach())
        del pairs, faith, open_cost, edge_cost, sparse_term

        # and the complement should be at chance: cross-entropy against a uniform
        # target, which is minimized when the reversed mask knows nothing
        reversed_pairs = logit_pairs(adapter, task, chunk, gates, originals, temperature, reverse=True,
                                edge_logits=edge_logits, edge_ids=edge_ids, noise=noise,
                                units=units)
        complete = functional.cross_entropy(
            reversed_pairs, torch.full_like(reversed_pairs, 0.5))
        (completeness * complete).backward()
        complete_value = float(complete.detach())
        del reversed_pairs, complete

        # A single non-finite gradient is not survivable: AdamW carries it into
        # both moments, every logit it touches becomes NaN on the next step,
        # `logits > 0` is False everywhere and the density reads 0.0 -- which is
        # how a 1.7B edge run reached step 25 with every edge shut and a NaN
        # faith it never came back from. The complement pass is the source. With
        # the mask reversed at high density a destination reads a residual its
        # own writes exactly cancel, and RMSNorm's backward at an exactly-zero
        # input is 0/0. Checking the loss does not catch it: the forward is
        # finite (all-edges-shut logits reach 151 with no NaN on Qwen3-1.7B),
        # so this checks what is about to be stepped instead. Zeroed rather than
        # skipped, so the step still carries every gate whose gradient is real.
        poisoned = 0
        for tensor in ([*gates.values(), *(units.parameters() if units else [])]
                       + ([edge_logits] if edge_logits is not None else [])):
            if tensor.grad is not None and not bool(torch.isfinite(tensor.grad).all()):
                poisoned += int((~torch.isfinite(tensor.grad)).sum())
                tensor.grad = torch.nan_to_num(tensor.grad, nan=0.0, posinf=0.0, neginf=0.0)
        optimizer.step()
        # Every step rather than once: AdamW's decay would walk a pinned logit
        # from 10 toward the threshold over a long run with no gradient to stop it.
        if pinned:
            pin(gates, pinned)
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            history.append({
                "step": step,
                "loss": round(faith_value + price * open_value + completeness * complete_value, 4),
                "faith": round(faith_value, 4),
                "open": round(open_value, 4),
                "complete": round(complete_value, 4),
                "price": round(price, 2),
                **({"edge_open": round(edge_value, 4),
                    "edge_price": round(edge_price, 3)} if edge_ids else {}),
            })
        if journal is not None:
            # Every step, because these three floats were already synced off the
            # device to build `history` and writing them costs a line of JSON.
            # `history` keeps its six entries -- it is the summary that ships
            # inside the artifact, and a 2000-row curve does not belong there.
            row = {
                "loss": faith_value + price * open_value + completeness * complete_value,
                "faith": faith_value, "open": open_value, "complete": complete_value,
                "price": price, "noise": noise,
            }
            if poisoned:
                # Never expected, and silent if it is not written down: a run
                # whose gradients are half NaN scores like one whose gates
                # simply stopped moving.
                row["poisoned"] = poisoned
            if multipliers is not None:
                row["target"] = goal
                row["density"] = hard_density
                row["lambda1"], row["lambda2"] = (float(v) for v in multipliers.detach())
                row["restart"] = int(restarted)
            if edge_ids:
                row["edge_open"] = edge_value
                row["edge_price"] = edge_price
                with torch.no_grad():
                    row["edge_density"] = float((edge_logits > 0).float().mean())
            # `open` is the relaxed cost the optimizer sees; `density` is the
            # hard fraction the result is quoted as, and the two come apart
            # exactly where the gates sit near zero. Throttled because it reduces
            # over every gate -- 1.4e9 of them on a whole 1.7B model.
            if probe_every and (step % probe_every == 0 or step == steps - 1):
                with torch.no_grad():
                    if "density" not in row:
                        row["density"] = (float((edge_logits > 0).float().mean())
                                          if not gate_weights else hard_count() / total)
                    if units is not None:
                        for family, entry in units.counts().items():
                            row[f"{family}s_open"] = entry["open"]
                    # The mask the result is quoted on, scored the way the result
                    # scores it: thresholded, no noise, held-out rows. `faith`
                    # above is a sampled mask's, and the two are different
                    # networks wherever the logits sit near zero. On the 1.7B a
                    # run held `faith` near 0 through its whole second half while
                    # this number sat at chance -- 60% of the gates had ended
                    # within 0.5 of the threshold, open on half the sampled
                    # passes and shut on every deterministic one -- and nothing
                    # else in the loop could see that, so it was found after
                    # 1h49m rather than at step 300.
                    row["hard_accuracy"] = ranking_accuracy(logit_pairs(
                        adapter, task, probe_rows, gates, originals, temperature,
                        deterministic=True, edge_logits=edge_logits, edge_ids=edge_ids,
                        units=units))
                    # Better means more accurate, or as accurate and sparser --
                    # which is what walking a density curve is for. Snapshotted
                    # to the CPU as one bool per gate: 1.4 GiB of host RAM on a
                    # 1.7B against holding a second copy on the device, and it
                    # only happens on an improvement.
                    here = row.get("density")
                    if here is None:
                        here = hard_count() / total
                    better = best["accuracy"] is None or (
                        row["hard_accuracy"] > best["accuracy"]
                        or (row["hard_accuracy"] >= best["accuracy"] and here < best["density"]))
                    if better:
                        best.update(accuracy=row["hard_accuracy"], density=here, step=step,
                                    gates={name: hard_open(name, logits).to("cpu")
                                           for name, logits in gates.items()},
                                    edges=(edge_logits.detach() > 0).to("cpu")
                                    if edge_logits is not None else None)
            journal.log(step, **row)

    with torch.no_grad():
        # Counted in integers: summed as float32 the count rounds past 2^24,
        # and every 1.7B artifact before this recorded an `open` a few gates
        # off the mask it sat beside.
        open_count = hard_count()
        final = logit_pairs(adapter, task, test_rows, gates, originals, temperature,
                       deterministic=True, edge_logits=edge_logits, edge_ids=edge_ids,
                       units=units)
        complement = logit_pairs(adapter, task, test_rows, gates, originals, temperature,
                            reverse=True, deterministic=True,
                            edge_logits=edge_logits, edge_ids=edge_ids, units=units)
        trained = ranking_accuracy(logit_pairs(adapter, task, train_rows, gates, originals, temperature,
                                   deterministic=True, edge_logits=edge_logits, edge_ids=edge_ids,
                                   units=units))
        # With units the mask is a product and no one logit says whether a
        # weight is open, so the effective boolean mask is what is returned.
        mask = ({name: logits.detach() for name, logits in gates.items()} if units is None
                else {name: hard_open(name, logits) for name, logits in gates.items()})
        # The question the ranking does not ask. One forward pass over the same
        # held-out rows, argmax over the whole vocabulary: a mask that keeps two
        # logits in order while the rest of the distribution collapses scores
        # 0.72 above and near zero here, and that gap is the whole reason the
        # generations read ' mind mind mind'.
        probe = {"chunk": max(1, batch), "gates": gates, "originals": originals,
                 "temperature": temperature, "deterministic": True,
                 "edge_logits": edge_logits, "edge_ids": edge_ids, "units": units}
        first = first_token_accuracy(adapter, task, test_rows, **probe)
        # Rescored on every held-out row rather than trusted from the probe, so
        # `best_accuracy` and `accuracy` are the same measurement of two
        # different masks instead of two measurements of two masks.
        if best["gates"] is not None or best["edges"] is not None:
            snapshot = ({name: tensor.to(device) for name, tensor in best["gates"].items()}
                        if best["gates"] else None)
            snap_edges = best["edges"].to(device) if best["edges"] is not None else None
            best["accuracy"] = ranking_accuracy(logit_pairs(
                adapter, task, test_rows, snapshot, originals, temperature,
                deterministic=True, edge_logits=snap_edges, edge_ids=edge_ids))
        base_first = first_token_accuracy(adapter, task, test_rows, chunk=max(1, batch),
                                  gates=None, originals=originals,
                                  temperature=temperature, deterministic=True)
    for name, parameter in targets.items():
        parameter.data.copy_(originals[name])
    return Sheaf(
        gates=mask,
        units=None if units is None else units.report(),
        unit_logits=None if units is None else units.state(),
        density=open_count / total, n_parameters=total, n_open=open_count,
        accuracy=ranking_accuracy(final), train_accuracy=trained,
        complement_accuracy=ranking_accuracy(complement),
        baseline_accuracy=baseline, history=history,
        layers=None if layers is None else sorted(layers),
        n_edges=len(edge_ids),
        n_edges_open=int((edge_logits > 0).sum()) if edge_ids else 0,
        edge_density=(float((edge_logits > 0).float().mean()) if edge_ids else None),
        edges=([edge for edge, keep in zip(edge_ids, (edge_logits > 0).tolist(), strict=True) if keep]
               if edge_ids else None),
        first_token=first, baseline_first_token=base_first,
        best_gates=best["gates"],
        best_edges=([edge for edge, keep in zip(edge_ids, best["edges"].tolist(), strict=True)
                     if keep] if best["edges"] is not None and edge_ids else None),
        best_accuracy=best["accuracy"],
        best_density=best["density"], best_step=best["step"],
        n_pinned=sum(int(indices.numel()) for indices in pinned.values()),
        n_restarts=restarts,
    )
