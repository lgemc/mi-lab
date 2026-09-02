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

    def __str__(self) -> str:
        return (f"sheaf: {self.density:.2%} of weights open · held-out {self.accuracy:.3f} "
                f"(train {self.train_accuracy:.3f}) against {self.baseline_accuracy:.3f} full · "
                f"complement {self.complement_accuracy:.3f}")

def _targets(adapter) -> Dict[str, torch.nn.Parameter]:
    """Every weight a gate is attached to: the blocks, minus norms and embeddings

    Norms and embeddings are excluded the way DiscoGP excludes them. A gated
    embedding deletes tokens rather than computation, and a gated norm changes
    what every surviving component reads, so neither is a statement about where
    the behaviour lives.
    """
    inside = {id(parameter) for block in adapter.blocks for name, parameter in block.named_parameters()
              if "ln" not in name and "norm" not in name}
    return {name: parameter for name, parameter in adapter.model.named_parameters()
            if id(parameter) in inside}

def _pairs(adapter, task: CircuitTask, rows: Sequence[int], gates: Optional[Dict[str, torch.Tensor]],
           originals: Dict[str, torch.Tensor], temperature: float, reverse: bool = False) -> torch.Tensor:
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
    encoded = adapter.tokenizer(list(prompts), return_tensors="pt", padding=True)
    adapter.tokenizer.padding_side = "right"
    ids = encoded["input_ids"].to(adapter.model.device)
    mask = encoded["attention_mask"].to(adapter.model.device)

    parameters = dict(adapter.model.named_parameters())
    if gates is not None:
        for name, logits in gates.items():
            sampled = gumbel_sigmoid(logits, temperature)
            parameters[name] = (1.0 - sampled if reverse else sampled) * originals[name]
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
          init: float = 1.0, batch: int = 8, max_times: float = 1000.0,
          warmup: Optional[int] = None, holdout: float = 0.25) -> Sheaf:
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
    """
    adapter = require_circuits(adapter)
    if steps < 1:
        raise CircuitError(f"pruning needs at least one step, got {steps}")
    targets = _targets(adapter)
    if not targets:
        raise CircuitError("no maskable weights found; every block was norms and embeddings")

    originals = {name: parameter.detach().clone() for name, parameter in targets.items()}
    for parameter in targets.values():
        parameter.requires_grad_(False)
    gates = {name: torch.full_like(parameter, init).requires_grad_(True)
             for name, parameter in targets.items()}
    total = sum(g.numel() for g in gates.values())
    optimizer = torch.optim.AdamW(list(gates.values()), lr=rate)

    # a mask trained and scored on the same prompts memorizes them. The first run
    # of this reported 0.02% of weights open at accuracy 1.000, which is not a
    # circuit, it is eight examples learned by 20k weights. DiscoGP splits; so does
    # this, and `accuracy` below is the held-out number.
    count = len(list(task.clean))
    split = max(1, int(count * (1.0 - holdout)))
    if split >= count:
        raise CircuitError(
            f"a task of {count} prompts leaves no holdout at {holdout:.0%}; a mask scored on its own "
            "training prompts reports memorization as faithfulness"
        )
    train_rows, test_rows = list(range(split)), list(range(split, count))
    with torch.no_grad():
        baseline = _accuracy(_pairs(adapter, task, test_rows, None, originals, temperature))

    warmup = steps if warmup is None else warmup
    history = []
    for step in range(steps):
        price = schedule(step, sparsity, max_times, warmup)
        chunk = train_rows[: batch] if batch else train_rows
        pairs = _pairs(adapter, task, chunk, gates, originals, temperature)
        # the masked model should prefer the right answer
        faith = functional.cross_entropy(pairs, torch.zeros(pairs.shape[0], dtype=torch.long,
                                                            device=pairs.device))
        # every open gate costs something, measured on the relaxed probability so
        # the term has a gradient where the hard gate does not
        open_cost = sum(torch.sigmoid(g).sum() for g in gates.values()) / total
        # and the complement should be at chance: cross-entropy against a uniform
        # target, which is minimized when the reversed mask knows nothing
        reversed_pairs = _pairs(adapter, task, chunk, gates, originals, temperature, reverse=True)
        complete = functional.cross_entropy(
            reversed_pairs, torch.full_like(reversed_pairs, 0.5))
        loss = faith + price * open_cost + completeness * complete
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, steps // 6) == 0 or step == steps - 1:
            history.append({
                "step": step, "loss": round(float(loss.detach()), 4),
                "faith": round(float(faith.detach()), 4),
                "open": round(float(open_cost.detach()), 4),
                "complete": round(float(complete.detach()), 4),
                "price": round(price, 2),
            })

    with torch.no_grad():
        kept = {name: (logits > 0).float() for name, logits in gates.items()}
        open_count = int(sum(k.sum() for k in kept.values()))
        final = _pairs(adapter, task, test_rows, gates, originals, temperature)
        complement = _pairs(adapter, task, test_rows, gates, originals, temperature, reverse=True)
        trained = _accuracy(_pairs(adapter, task, train_rows, gates, originals, temperature))
    for name, parameter in targets.items():
        parameter.data.copy_(originals[name])
    return Sheaf(
        gates={name: logits.detach() for name, logits in gates.items()},
        density=open_count / total, n_parameters=total, n_open=open_count,
        accuracy=_accuracy(final), train_accuracy=trained,
        complement_accuracy=_accuracy(complement),
        baseline_accuracy=baseline, history=history,
    )
