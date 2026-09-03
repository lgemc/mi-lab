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
half that the node-level `mask` technique in discovery.py cannot express.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as functional

from ..data.tasks import CircuitTask
from ..model.adapter import require_circuits
from ..telemetry.journal import Journal
from .circuits import CircuitError


def gumbel_sigmoid(logits: torch.Tensor, temperature: float = 1.0, eps: float = 1e-10,
                   noise: float = 1.0) -> torch.Tensor:
    """A Bernoulli gate that is hard in the forward pass and differentiable in the backward one

    Straight-through: the returned value is exactly 0 or 1, so the model really
    runs with the gate shut, while the gradient sees the relaxed sigmoid and can
    move the logit that produced it. Rounding without the straight-through trick
    gives a zero gradient everywhere; relaxing without the rounding trains a
    model that is never actually pruned and scores well because a half-open gate
    still passes half the signal.

    The paper's eq. 3: `sigma((l - log(log U1 / log U2)) / tau)`, logistic noise
    added *before* the division. So `temperature` never changes which gates
    open -- `sigma(x / tau) > 0.5` is `x > 0` at any tau -- it only sharpens the
    backward pass: at the reference's 0.01 for weight masks a gate on the
    boundary gets a hundred times the gradient and a gate away from it none.
    Annealing tau, the Gumbel-softmax recipe, therefore cannot bring the sample
    to the thresholded mask that is evaluated; `noise` is the scale that can.
    At 1.0 the gate is the paper's; at 0.0 it is exactly `logits > 0`.
    """
    relaxed_input = logits
    if noise > 0.0:
        uniform = logits.new_empty([2, *logits.shape]).uniform_(0, 1)
        drawn = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
        relaxed_input = logits + noise * drawn
    relaxed = torch.sigmoid(relaxed_input / temperature)
    return ((relaxed > 0.5).type_as(relaxed) - relaxed).detach() + relaxed

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
    # Gates held open outside the search (`protect`); inside `n_open`.
    n_pinned: int = 0
    # Steps on which the learned price was reset because the constraint held.
    n_restarts: int = 0

    def __str__(self) -> str:
        # The band is in the string because `density` is a fraction of what was
        # gated, not of the model: 1% open across seven layers and 1% open
        # across twenty-eight are different claims wearing the same number.
        band = "all layers" if self.layers is None else f"layers {span(self.layers)}"
        edges = ("" if self.edge_density is None
                 else f" · {self.edge_density:.2%} of {self.n_edges} edges")
        return (f"sheaf: {self.density:.2%} of {band} open{edges} · "
                f"held-out {self.accuracy:.3f} "
                f"(train {self.train_accuracy:.3f}) against {self.baseline_accuracy:.3f} full · "
                f"complement {self.complement_accuracy:.3f}")

def span(layers: Sequence[int]) -> str:
    """`21-27` for a contiguous band, the list itself for anything else"""
    ordered = sorted(layers)
    if len(ordered) > 1 and ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}-{ordered[-1]}"
    return ",".join(str(layer) for layer in ordered)

def gateable(adapter, layers: Optional[Sequence[int]] = None) -> Dict[str, torch.nn.Parameter]:
    """Every weight a gate is attached to: the blocks, minus norms and embeddings

    Norms and embeddings are excluded the way DiscoGP excludes them. A gated
    embedding deletes tokens rather than computation, and a gated norm changes
    what every surviving component reads, so neither is a statement about where
    the behaviour lives.

    `layers` narrows that to a band, and the reason is arithmetic rather than
    method. Every gate carries a float32 logit, its gradient and two AdamW
    moments, so gating all of a 1.7B model's 1.41B block weights costs ~23 GiB
    of optimizer state before a single activation is stored -- where GPT-2
    small's 85M gates cost 1.4 GiB, which is why this never came up. A band
    spends less and says so: the claim it supports is "within these layers",
    which is a smaller claim and is written into the artifact as one. Reaching
    for a leaner optimizer instead would buy the same memory by making the
    scope of the result harder to see rather than easier.
    """
    count = len(adapter.blocks)
    chosen = list(range(count)) if layers is None else list(layers)
    outside = [layer for layer in chosen if not 0 <= layer < count]
    if outside:
        raise CircuitError(
            f"layers {outside} are not in this model, which has {count} blocks (0-{count - 1})"
        )
    if not chosen:
        raise CircuitError("an empty layer band gates nothing; omit `layers` to gate the whole model")
    inside = {id(parameter) for layer in chosen
              for name, parameter in adapter.blocks[layer].named_parameters()
              if "ln" not in name and "norm" not in name}
    return {name: parameter for name, parameter in adapter.model.named_parameters()
            if id(parameter) in inside}

def _pairs(adapter, task: CircuitTask, rows: Sequence[int], gates: Optional[Dict[str, torch.Tensor]],
           originals: Dict[str, torch.Tensor], temperature: float, reverse: bool = False,
           deterministic: bool = False,
           edge_logits: Optional[torch.Tensor] = None,
           edge_ids: Optional[Sequence[tuple]] = None,
           whole: bool = False, noise: float = 1.0) -> torch.Tensor:
    """The good/bad logit pair at each named prompt's last real token, under the current gates

    `whole` returns the entire last-token distribution instead of two of it,
    which is what a KL faithfulness term needs. Two logits are a ranking; the
    distribution is what the model would actually say.

    Indexed by row rather than handed a list of prompts, because the answers
    live in a parallel array and slicing one without the other scores every
    prompt against another prompt's names. A first version took prompts
    directly and only happened to be right because the slice was a prefix.
    """
    every = list(task.clean)
    prompts = [every[row] for row in rows]
    io_all, subject_all = task.answers(adapter)
    io = [io_all[row] for row in rows]
    subject = [subject_all[row] for row in rows]
    # Set before encoding, not after. The backend's `_encode` mutates this on the
    # shared tokenizer and leaves it "left" after any generation, and the last
    # real token is found below as `mask.sum(dim=1) - 1`, which is a pad position
    # under left padding. Setting it afterwards fixes the *next* call and scores
    # this one on whatever the previous caller happened to leave behind.
    adapter.tokenizer.padding_side = "right"
    encoded = adapter.tokenizer(list(prompts), return_tensors="pt", padding=True)
    ids = encoded["input_ids"].to(adapter.model.device)
    mask = encoded["attention_mask"].to(adapter.model.device)

    parameters = dict(adapter.model.named_parameters())
    if gates is not None:
        for name, logits in gates.items():
            # Training samples; evaluation thresholds. Scoring a sampled mask
            # measures a different random subnetwork on every forward pass, and
            # it is not the mask that was learned: a first version sampled
            # everywhere and reported train accuracy flat at 0.70 from 55% of
            # the weights down to 0.02%, because the number never depended on
            # the gates at all.
            sampled = ((logits > 0).to(logits.dtype) if deterministic
                       else gumbel_sigmoid(logits, temperature, noise=noise))
            # The gate logits are float32 whatever the model's dtype is (see
            # `prune`), so the mask is cast down to the weight rather than the
            # weight promoted up to the mask: promoting leaves this one
            # parameter in float32 while every activation reaching it is
            # bfloat16, which is a dtype error on a good day and a silent
            # upcast of one matmul on a bad one.
            gate = (1.0 - sampled if reverse else sampled).to(originals[name].dtype)
            parameters[name] = gate * originals[name]
    # The edge hooks sit on the modules and fire inside functional_call, which
    # swaps parameters and leaves hooks alone -- so a head's write is
    # reconstructed from the *masked* weights, and the two halves compose
    # rather than each measuring the unmasked model.
    # Sampled here rather than handed in, for the same reason the weight gates
    # are: the faith pass and the complement pass each call backward, and a
    # sample hoisted out of both would have its graph freed by the first.
    edges = None
    if edge_logits is not None and edge_ids:
        drawn = ((edge_logits > 0).to(edge_logits.dtype) if deterministic
                 else gumbel_sigmoid(edge_logits, temperature, noise=noise))
        if reverse:
            drawn = 1.0 - drawn
        edges = {edge: drawn[index] for index, edge in enumerate(edge_ids)}
    with adapter.edge_gate(edges) if edges else _nothing():
        logits = torch.func.functional_call(
            adapter.model, {**parameters, **dict(adapter.model.named_buffers())},
            (ids,), {"attention_mask": mask, "use_cache": False},
        ).logits
    last = mask.sum(dim=1) - 1
    index = torch.arange(logits.shape[0], device=logits.device)
    final = logits[index, last]
    if whole:
        return final
    good = final[index, torch.tensor(io, device=logits.device)]
    bad = final[index, torch.tensor(subject, device=logits.device)]
    return torch.stack([good, bad], dim=-1)

@contextmanager
def _nothing():
    """A do-nothing context, so the forward is written once rather than twice"""
    yield

def _accuracy(pairs: torch.Tensor) -> float:
    return float((pairs[:, 0] > pairs[:, 1]).float().mean())

def _split(prompts: Sequence[str], holdout: float) -> "tuple[List[int], List[int]]":
    """Train/held-out rows, split so that no *prompt* lands on both sides

    This was `range(split), range(split, count)` -- a split by row index, which
    is only a holdout when every row is a different prompt. The registered
    translation task draws `size` examples from a pool with replacement and its
    clean prompt depends on one word, so on this model it has 25 distinct
    prompts however many rows are asked for: at size 128 the 96/32 row split put
    all 32 held-out rows on prompts the mask had trained on, and `accuracy` --
    the number that exists specifically to be the honest one -- was measuring
    memorization for the third time in this module's history.

    Grouping by the prompt is what `LabeledPrompts.split` already does for a
    contrast pair, and for the same reason: a group straddling the split makes
    the metric about the thing that was supposed to be held back.
    """
    order: Dict[str, List[int]] = {}
    for row, prompt in enumerate(prompts):
        order.setdefault(prompt, []).append(row)
    groups = list(order.values())
    kept = max(1, int(len(groups) * (1.0 - holdout)))
    if kept >= len(groups):
        raise CircuitError(
            f"{len(prompts)} rows carry only {len(groups)} distinct prompts, which leaves no "
            f"holdout at {holdout:.0%} once rows sharing a prompt are kept together. A mask "
            f"scored on prompts it trained on reports memorization as faithfulness -- widen the "
            f"task's pool rather than its row count, because repeating a prompt adds rows and "
            f"no information."
        )
    train = [row for group in groups[:kept] for row in group]
    test = [row for group in groups[kept:] for row in group]
    return sorted(train), sorted(test)

def _chunk(train_rows: Sequence[int], step: int, batch: int) -> List[int]:
    """The batch this step trains on, walked round the training rows

    This used to be `train_rows[:batch]`, which is the same eight prompts on
    every step of every run: `size` grew the holdout and never reached the
    training set at all. The retracted run in 5cc8dc3 was described as 24
    training examples and was really 8, repeated 500 times, and "the data
    regime is wrong" was the conclusion drawn from it -- so the standing
    explanation for why the method does not reproduce here was measuring this
    line rather than the method. A stride keeps the pass deterministic, which
    a shuffle would not, while still showing the optimizer every example.
    """
    rows = list(train_rows)
    if not batch or batch >= len(rows):
        return rows
    start = (step * batch) % len(rows)
    return [rows[(start + offset) % len(rows)] for offset in range(batch)]

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
            "open": _accuracy(_pairs(adapter, task, chosen, None, originals, 1.0,
                                     deterministic=True)),
            "shut": _accuracy(_pairs(adapter, task, chosen, shut, originals, 1.0,
                                     deterministic=True)),
        }

def schedule(step: int, lambda_0: float, max_times: float, warmup: int) -> float:
    """Ramp the sparsity price from lambda_0 to lambda_0 * max_times over `warmup` steps

    A constant price does not work and the reference's defaults say why: it
    ramps to a thousand times its starting value. Early on the gates have to
    find which weights the task needs, and a price high enough to close them
    all drowns that out; late on nothing else will push a working subnetwork to
    give up the weights it does not need. Holding it constant at any value
    picks one of those failures.
    """
    if warmup <= 0 or step >= warmup:
        return lambda_0 * max_times
    return lambda_0 + lambda_0 * (max_times - 1.0) * step / warmup

def target_schedule(step: int, target: float, warmup: int) -> float:
    """Lower the density target from 1.0 to `target` over `warmup` steps, then hold it

    CoFi's schedule (2204.00408): the constraint starts satisfied and tightens,
    so the learned price never has to spike to catch up with a target the mask
    is nowhere near. Held afterwards, because a target reached on the last
    step is a mask that was never trained at its own density.
    """
    if warmup <= 0 or step >= warmup:
        return target
    return 1.0 - (1.0 - target) * step / warmup

# A pinned gate's logit: sigmoid(10) samples open 99.995% of the time, and the
# threshold reads it as open, so nothing downstream has to know it is pinned.
PINNED = 10.0

def protected(originals: Dict[str, torch.Tensor], fraction: float) -> Dict[str, torch.Tensor]:
    """The flat indices, per tensor, of the top `fraction` of weights by magnitude

    The top 0.01% by |w| of Qwen3-1.7B, 150k weights of 1.5B, are the ones
    without which it says ` the the the`: removing them costs 8 nats where a
    random 4% of the weights costs nothing (scratchpad fragility test, 2026-
    09-03). Not the only such set: the top 0.1% by Wanda's |w| * ||x|| (Sun et
    al. 2023) overlaps it 7% and costs as much, on a layer-2 down_proj input
    feature at 77,000x the median norm. Magnitude is the half that needs no
    forward pass. Ranked over every gated tensor at once rather than within each,
    because that is where the weights are: a per-tensor cut would pin the
    biggest weights of a tensor that has none.
    """
    if fraction <= 0:
        return {}
    magnitudes = torch.cat([w.detach().abs().float().flatten() for w in originals.values()])
    count = max(1, int(fraction * magnitudes.numel()))
    threshold = torch.topk(magnitudes, count).values[-1]
    del magnitudes
    return {name: (w.detach().abs().float().flatten() >= threshold).nonzero().flatten()
            for name, w in originals.items()}

def pin(gates: Dict[str, torch.Tensor], pinned: Dict[str, torch.Tensor]) -> None:
    """Hold every protected gate at `PINNED`, after the optimizer has moved the rest"""
    with torch.no_grad():
        for name, indices in pinned.items():
            if indices.numel():
                gates[name].view(-1)[indices] = PINNED

def prune(adapter, task: CircuitTask, steps: int = 500, rate: float = 0.1,
          sparsity: float = 1.0, completeness: float = 0.3, temperature: float = 1.0,
          init: float = 1.0, batch: int = 64, max_times: float = 1000.0,
          warmup: Optional[int] = None, holdout: float = 0.25,
          layers: Optional[Sequence[int]] = None,
          journal: Optional[Journal] = None, probe_every: int = 10,
          seed: Optional[int] = None, edge_sparsity: float = 0.0,
          faith_kind: str = "pair", anneal: bool = False,
          target: Optional[float] = None, protect: float = 0.0,
          dual_rate: Optional[float] = None, dual_restart: bool = False) -> Sheaf:
    """Learn a weight mask that does the task and whose complement cannot

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
    `probe_every` throttles the two metrics that are not free: `density`
    reduces over every gate, and `hard_accuracy` is a forward pass of the
    *thresholded* mask on held-out rows -- the mask the run is quoted on,
    which the loss terms never touch because they are measured on sampled
    ones. That gap is where a run collapses without the curve saying so.
    """
    adapter = require_circuits(adapter)
    if steps < 1:
        raise CircuitError(f"pruning needs at least one step, got {steps}")
    if seed is not None:
        # Global rather than a threaded Generator: gumbel_sigmoid is called once
        # per gated parameter per pass and a generator argument would have to
        # reach every one of them. A process runs one prune, so the global state
        # is not shared with anything that would notice.
        torch.manual_seed(seed)
    targets = gateable(adapter, layers)
    if not targets:
        raise CircuitError("no maskable weights found; every block was norms and embeddings")

    originals = {name: parameter.detach().clone() for name, parameter in targets.items()}
    for parameter in targets.values():
        parameter.requires_grad_(False)
    # float32 whatever the weights are. `full_like` inherited the model's dtype,
    # which is bfloat16 on every model here larger than GPT-2, and bfloat16
    # carries eight mantissa bits: AdamW's second moment underflows, and
    # `open_cost` below sums sigmoid over every gate, where accumulating 1.4e9
    # terms in bfloat16 stops adding once the running total passes 256.
    gates = {name: torch.full(parameter.shape, init, dtype=torch.float32,
                              device=parameter.device).requires_grad_(True)
             for name, parameter in targets.items()}
    total = sum(g.numel() for g in gates.values())
    pinned = protected(originals, protect)
    pin(gates, pinned)
    optimizer = torch.optim.AdamW(list(gates.values()), lr=rate)

    # a mask trained and scored on the same prompts memorizes them. The first run
    # of this reported 0.02% of weights open at accuracy 1.000, which is not a
    # circuit, it is eight examples learned by 20k weights. DiscoGP splits; so does
    # this, and `accuracy` below is the held-out number.
    train_rows, test_rows = _split(list(task.clean), holdout)
    with torch.no_grad():
        baseline = _accuracy(_pairs(adapter, task, test_rows, None, originals, temperature,
                                    deterministic=True))
    # A fixed handful of held-out rows for the probe below, the same rows every
    # time so the curve is one quantity over the run rather than one per draw.
    probe_rows = test_rows[:max(1, min(len(test_rows), batch))]

    # One scalar per (source, destination) the residual stream admits. Cheap
    # next to the weights -- 2028 against 85M on GPT-2 small -- and the half
    # that says which paths exist rather than how strong they are.
    edge_ids = list(adapter.edges()) if edge_sparsity > 0 else []
    edge_logits = None
    if edge_ids:
        device = next(iter(gates.values())).device
        edge_logits = torch.full((len(edge_ids),), init, dtype=torch.float32,
                                 device=device).requires_grad_(True)
        optimizer.add_param_group({"params": [edge_logits]})

    if faith_kind not in ("pair", "kl", "nll", "gold"):
        raise CircuitError(
            f"unknown faith_kind '{faith_kind}'; known kinds are 'nll' (the paper's), 'kl', "
            "'gold' (nll on the task's answer) and 'pair' (this repo's original, and too "
            "weak -- see the docstring)")
    # The reference distribution, taken once from the unmasked model. The
    # weights are frozen, so what the full model says never changes and
    # recomputing it every step would only pay for the same numbers again.
    reference = None
    if faith_kind == "gold":
        # The label is the task's, and the full model is never consulted.
        io, _ = task.answers(adapter)
        device = next(iter(gates.values())).device
        reference = {row: torch.tensor([io[row]], device=device) for row in train_rows}
    if faith_kind in ("kl", "nll"):
        with torch.no_grad():
            reference = {}
            for row in train_rows:
                full = _pairs(adapter, task, [row], None, originals, temperature,
                              deterministic=True, whole=True)
                # KL wants the distribution; NLL wants the token the full model
                # would actually emit, which is what the paper's y-hat is.
                reference[row] = (full.log_softmax(dim=-1) if faith_kind == "kl"
                                  else full.argmax(dim=-1))

    if target is not None and not 0.0 < target <= 1.0:
        raise CircuitError(f"a density target is a fraction in (0, 1], got {target}")
    if warmup is None:
        warmup = steps // 2 if target is not None else steps
    # The learned price. Two scalars, ascended rather than descended: their
    # gradient is negated before the step below, so one AdamW on both sides
    # of the saddle. Started at zero, so step 0 is faith alone.
    multipliers = None
    if target is not None:
        device = next(iter(gates.values())).device
        multipliers = torch.zeros(2, dtype=torch.float32, device=device)
        if dual_rate is None:
            multipliers.requires_grad_(True)
            optimizer.add_param_group({"params": [multipliers], "weight_decay": 0.0})
    elif dual_rate is not None or dual_restart:
        raise CircuitError("dual_rate and dual_restart shape the learned price: give a target")
    restarts = 0
    history = []
    for step in range(steps):
        price = schedule(step, sparsity, max_times, warmup)
        goal = target_schedule(step, target, warmup) if target is not None else None
        noise = 1.0 - step / max(1, steps - 1) if anneal else 1.0
        chunk = _chunk(train_rows, step, batch)
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
        pairs = _pairs(adapter, task, chunk, gates, originals, temperature,
                       edge_logits=edge_logits, edge_ids=edge_ids,
                       whole=(faith_kind in ("kl", "nll", "gold")), noise=noise)
        if faith_kind in ("nll", "gold"):
            # The paper's term: -sum_i log p_m(y-hat_i | x_i), the likelihood the
            # masked model gives the token the *full* model predicted, over the
            # whole vocabulary. Not a choice between two candidates. "gold"
            # is the same sum with the task's y_i in place of y-hat_i.
            labels = torch.cat([reference[row] for row in chunk], dim=0)
            faith = functional.cross_entropy(pairs, labels)
        elif faith_kind == "kl":
            # The soft-target relative: the whole distribution rather than its
            # argmax. The paper uses KL to *evaluate* rather than to train.
            labels = torch.cat([reference[row] for row in chunk], dim=0)
            faith = functional.kl_div(pairs.log_softmax(dim=-1), labels,
                                      log_target=True, reduction="batchmean")
        else:
            # the masked model should prefer the right answer
            faith = functional.cross_entropy(pairs, torch.zeros(pairs.shape[0], dtype=torch.long,
                                                                device=pairs.device))
        # every open gate costs something, measured on the relaxed probability so
        # the term has a gradient where the hard gate does not
        open_cost = sum(torch.sigmoid(g).sum() for g in gates.values()) / total
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
                hard_density = float(sum((g > 0).sum() for g in gates.values()) / total)
            gap = hard_density - goal
            # The marginal price of an open gate under the Lagrangian,
            # `l1 + 2 l2 (density - t)`, applied to the relaxed cost so the
            # gates have a gradient. Detached: the multipliers are ascended
            # on the gap itself, below, not through this product.
            price = float((multipliers[0] + 2.0 * multipliers[1] * gap).detach())
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
        reversed_pairs = _pairs(adapter, task, chunk, gates, originals, temperature, reverse=True,
                                edge_logits=edge_logits, edge_ids=edge_ids, noise=noise)
        complete = functional.cross_entropy(
            reversed_pairs, torch.full_like(reversed_pairs, 0.5))
        (completeness * complete).backward()
        complete_value = float(complete.detach())
        del reversed_pairs, complete

        optimizer.step()
        # Every step rather than once: AdamW's decay would walk a pinned logit
        # from 10 toward the threshold over a long run with no gradient to stop it.
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
                        row["density"] = float(sum((g > 0).sum() for g in gates.values()) / total)
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
                    row["hard_accuracy"] = _accuracy(_pairs(
                        adapter, task, probe_rows, gates, originals, temperature,
                        deterministic=True, edge_logits=edge_logits, edge_ids=edge_ids))
            journal.log(step, **row)

    with torch.no_grad():
        # Counted in integers: summed as float32 the count rounds past 2^24,
        # and every 1.7B artifact before this recorded an `open` a few gates
        # off the mask it sat beside.
        open_count = int(sum(int((logits > 0).sum()) for logits in gates.values()))
        final = _pairs(adapter, task, test_rows, gates, originals, temperature,
                       deterministic=True, edge_logits=edge_logits, edge_ids=edge_ids)
        complement = _pairs(adapter, task, test_rows, gates, originals, temperature,
                            reverse=True, deterministic=True,
                            edge_logits=edge_logits, edge_ids=edge_ids)
        trained = _accuracy(_pairs(adapter, task, train_rows, gates, originals, temperature,
                                   deterministic=True, edge_logits=edge_logits, edge_ids=edge_ids))
    for name, parameter in targets.items():
        parameter.data.copy_(originals[name])
    return Sheaf(
        gates={name: logits.detach() for name, logits in gates.items()},
        density=open_count / total, n_parameters=total, n_open=open_count,
        accuracy=_accuracy(final), train_accuracy=trained,
        complement_accuracy=_accuracy(complement),
        baseline_accuracy=baseline, history=history,
        layers=None if layers is None else sorted(layers),
        n_edges=len(edge_ids),
        n_edges_open=int((edge_logits > 0).sum()) if edge_ids else 0,
        edge_density=(float((edge_logits > 0).float().mean()) if edge_ids else None),
        n_pinned=sum(int(indices.numel()) for indices in pinned.values()),
        n_restarts=restarts,
    )
