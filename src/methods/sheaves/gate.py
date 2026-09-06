"""The gate itself: how it samples, what it costs, and which gates never move.

A gate is one number per weight and everything about training it that does not
need a model is here -- so the loop in `training` reads as the loop rather than
as the arithmetic of a Gumbel-sigmoid.

Four decisions, each of which has been the difference between a run and a
wasted afternoon:

    gumbel_sigmoid    hard forward, relaxed backward. A deterministic sigmoid
                      gate lets the optimizer sit at 0.5 forever, which is
                      neither open nor closed and scores well because half of
                      everything is still there. `noise` rather than
                      `temperature` is the knob that brings the sample to the
                      mask that is actually saved.
    schedule          the hand-ramped sparsity price. A constant price picks
                      one of two failures: high enough to close everything
                      early, or never enough to make a working subnetwork give
                      up what it does not need.
    target_schedule   the learned alternative (CoFi, 2204.00408): name a
                      density and let the price find itself, starting from a
                      constraint that is satisfied and tightening.
    protected / pin   the top fraction by |w|, held open. The top 0.01% of
                      Qwen3-1.7B are the weights without which it says
                      ` the the the`.

A common pipe could be: gumbel_sigmoid | schedule | pin
"""

from typing import Dict

import torch


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
