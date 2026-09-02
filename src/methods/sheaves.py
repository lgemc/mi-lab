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

    faith       the masked model still prefers the right answer
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

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as functional

from ..data.tasks import CircuitTask
from ..model.adapter import require_circuits
from ..telemetry.journal import Journal
from .circuits import CircuitError


def gumbel_sigmoid(logits: torch.Tensor, temperature: float = 1.0, eps: float = 1e-10) -> torch.Tensor:
    """A Bernoulli gate that is hard in the forward pass and differentiable in the backward one

    Straight-through: the returned value is exactly 0 or 1, so the model really
    runs with the gate shut, while the gradient sees the relaxed sigmoid and can
    move the logit that produced it. Rounding without the straight-through trick
    gives a zero gradient everywhere; relaxing without the rounding trains a
    model that is never actually pruned and scores well because a half-open gate
    still passes half the signal.
    """
    uniform = logits.new_empty([2, *logits.shape]).uniform_(0, 1)
    noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
    relaxed = torch.sigmoid((logits + noise) / temperature)
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

    def __str__(self) -> str:
        # The band is in the string because `density` is a fraction of what was
        # gated, not of the model: 1% open across seven layers and 1% open
        # across twenty-eight are different claims wearing the same number.
        band = "all layers" if self.layers is None else f"layers {span(self.layers)}"
        return (f"sheaf: {self.density:.2%} of {band} open · held-out {self.accuracy:.3f} "
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
           deterministic: bool = False) -> torch.Tensor:
    """The good/bad logit pair at each named prompt's last real token, under the current gates

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
                       else gumbel_sigmoid(logits, temperature))
            # The gate logits are float32 whatever the model's dtype is (see
            # `prune`), so the mask is cast down to the weight rather than the
            # weight promoted up to the mask: promoting leaves this one
            # parameter in float32 while every activation reaching it is
            # bfloat16, which is a dtype error on a good day and a silent
            # upcast of one matmul on a bad one.
            gate = (1.0 - sampled if reverse else sampled).to(originals[name].dtype)
            parameters[name] = gate * originals[name]
    logits = torch.func.functional_call(
        adapter.model, {**parameters, **dict(adapter.model.named_buffers())},
        (ids,), {"attention_mask": mask, "use_cache": False},
    ).logits
    last = mask.sum(dim=1) - 1
    rows = torch.arange(logits.shape[0], device=logits.device)
    good = logits[rows, last, torch.tensor(io, device=logits.device)]
    bad = logits[rows, last, torch.tensor(subject, device=logits.device)]
    return torch.stack([good, bad], dim=-1)

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

def prune(adapter, task: CircuitTask, steps: int = 500, rate: float = 0.1,
          sparsity: float = 1.0, completeness: float = 0.3, temperature: float = 1.0,
          init: float = 1.0, batch: int = 64, max_times: float = 1000.0,
          warmup: Optional[int] = None, holdout: float = 0.25,
          layers: Optional[Sequence[int]] = None,
          journal: Optional[Journal] = None, probe_every: int = 10,
          seed: Optional[int] = None) -> Sheaf:
    """Learn a weight mask that does the task and whose complement cannot

    `init` starts every gate open -- a logit of 3 is a sigmoid of 0.95 -- so the
    search prunes a working model down rather than growing one from nothing.
    Starting closed makes the faithfulness term flat: a model with no weights
    has no gradient pointing at which weight to restore first.

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
    `probe_every` throttles the one metric that is not free: `density` reduces
    over every gate, where the loss terms were already synced off the device.
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
    optimizer = torch.optim.AdamW(list(gates.values()), lr=rate)

    # a mask trained and scored on the same prompts memorizes them. The first run
    # of this reported 0.02% of weights open at accuracy 1.000, which is not a
    # circuit, it is eight examples learned by 20k weights. DiscoGP splits; so does
    # this, and `accuracy` below is the held-out number.
    train_rows, test_rows = _split(list(task.clean), holdout)
    with torch.no_grad():
        baseline = _accuracy(_pairs(adapter, task, test_rows, None, originals, temperature,
                                    deterministic=True))

    warmup = steps if warmup is None else warmup
    history = []
    for step in range(steps):
        price = schedule(step, sparsity, max_times, warmup)
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
        pairs = _pairs(adapter, task, chunk, gates, originals, temperature)
        # the masked model should prefer the right answer
        faith = functional.cross_entropy(pairs, torch.zeros(pairs.shape[0], dtype=torch.long,
                                                            device=pairs.device))
        # every open gate costs something, measured on the relaxed probability so
        # the term has a gradient where the hard gate does not
        open_cost = sum(torch.sigmoid(g).sum() for g in gates.values()) / total
        (faith + price * open_cost).backward()
        faith_value, open_value = float(faith.detach()), float(open_cost.detach())
        del pairs, faith, open_cost

        # and the complement should be at chance: cross-entropy against a uniform
        # target, which is minimized when the reversed mask knows nothing
        reversed_pairs = _pairs(adapter, task, chunk, gates, originals, temperature, reverse=True)
        complete = functional.cross_entropy(
            reversed_pairs, torch.full_like(reversed_pairs, 0.5))
        (completeness * complete).backward()
        complete_value = float(complete.detach())
        del reversed_pairs, complete

        optimizer.step()
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            history.append({
                "step": step,
                "loss": round(faith_value + price * open_value + completeness * complete_value, 4),
                "faith": round(faith_value, 4),
                "open": round(open_value, 4),
                "complete": round(complete_value, 4),
                "price": round(price, 2),
            })
        if journal is not None:
            # Every step, because these three floats were already synced off the
            # device to build `history` and writing them costs a line of JSON.
            # `history` keeps its six entries -- it is the summary that ships
            # inside the artifact, and a 2000-row curve does not belong there.
            row = {
                "loss": faith_value + price * open_value + completeness * complete_value,
                "faith": faith_value, "open": open_value, "complete": complete_value,
                "price": price,
            }
            # `open` is the relaxed cost the optimizer sees; `density` is the
            # hard fraction the result is quoted as, and the two come apart
            # exactly where the gates sit near zero. Throttled because it reduces
            # over every gate -- 1.4e9 of them on a whole 1.7B model.
            if probe_every and (step % probe_every == 0 or step == steps - 1):
                with torch.no_grad():
                    row["density"] = float(sum((g > 0).sum() for g in gates.values()) / total)
            journal.log(step, **row)

    with torch.no_grad():
        kept = {name: (logits > 0).float() for name, logits in gates.items()}
        open_count = int(sum(k.sum() for k in kept.values()))
        final = _pairs(adapter, task, test_rows, gates, originals, temperature, deterministic=True)
        complement = _pairs(adapter, task, test_rows, gates, originals, temperature,
                            reverse=True, deterministic=True)
        trained = _accuracy(_pairs(adapter, task, train_rows, gates, originals, temperature,
                                   deterministic=True))
    for name, parameter in targets.items():
        parameter.data.copy_(originals[name])
    return Sheaf(
        gates={name: logits.detach() for name, logits in gates.items()},
        density=open_count / total, n_parameters=total, n_open=open_count,
        accuracy=_accuracy(final), train_accuracy=trained,
        complement_accuracy=_accuracy(complement),
        baseline_accuracy=baseline, history=history,
        layers=None if layers is None else sorted(layers),
    )
