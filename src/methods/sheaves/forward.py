"""Running the model under a mask: the forward pass, the loss terms, and the rows to run it on.

The training loop needs one forward pass that can be asked for several
different things -- two logits or the whole distribution, gates or no gates,
the mask or its complement, weight gates or edge gates or both -- and writing
that pass more than once is how two of them end up disagreeing about which
token is the last real one. So it is written once, in `logit_pairs`, and every
caller here and in `training` goes through it.

Weights are masked through `torch.func.functional_call` rather than copied into
the parameters in place. The reference implementation assigns masked values
onto frozen parameters with `copy_`; that is faster and it is also how a
gradient quietly stops flowing to the thing being trained, so this takes the
functional route and pays for it in memory.

Two scorings, and the difference between them is the difference between a
number that flatters and one that does not. `ranking_accuracy` asks whether the
answer beats one distractor, which two logits can satisfy while everything else
in the distribution outranks both; `first_token_accuracy` asks what generation
actually asks.

`split_rows` and `chunk_rows` are here rather than in the loop because both
have been silently wrong in this module's history and both are now tested
directly: a split by row index is not a holdout when rows repeat a prompt, and
`train_rows[:batch]` is the same eight examples on every step of every run.

A common pipe could be: split_rows | chunk_rows | logit_pairs | faith_term
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import torch

from ...data.tasks import CircuitTask
from ..common.errors import SheafError
from .faith import Faithfulness
from .gate import gumbel_sigmoid

if TYPE_CHECKING:
    from .units import Units

@contextmanager
def no_intervention():
    """A do-nothing context, so the forward is written once rather than twice"""
    yield

def logit_pairs(adapter, task: CircuitTask, rows: Sequence[int], gates: Optional[Dict[str, torch.Tensor]],
                originals: Dict[str, torch.Tensor], temperature: float, reverse: bool = False,
                deterministic: bool = False,
                edge_logits: Optional[torch.Tensor] = None,
                edge_ids: Optional[Sequence[tuple]] = None,
                whole: bool = False, noise: float = 1.0,
                units: Optional["Units"] = None,
                weights: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    """The good/bad logit pair at each named prompt's last real token, under the current gates

    `whole` returns the entire last-token distribution instead of two of it,
    which is what a KL faithfulness term needs. Two logits are a ranking; the
    distribution is what the model would actually say.

    `units` multiplies each weight's sampled gate by the sampled gates of the
    head, neuron and block it belongs to (see units.py); the unit gates are
    drawn once per call, so every slice of a head sees the same draw.
    `weights` substitutes tensors for the model's own, with no gates: it is
    how `attribution` runs the full model through leaves that carry a gradient.

    Indexed by row rather than handed a list of prompts, because the answers
    live in a parallel array and slicing one without the other scores every
    prompt against another prompt's names. A first version took prompts
    directly and only happened to be right because the slice was a prefix.
    """
    every = list(task.clean)
    prompts = [every[row] for row in rows]
    # The ids come off the task's readout rather than off the task, so the rule
    # that turns two logits into a number and the ids inside it are one object.
    score = task.readout(adapter)
    io = [score.positive[row] for row in rows]
    subject = [score.negative[row] for row in rows]
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
    if weights is not None:
        parameters.update(weights)
    # The unit gates are drawn once here and read per tensor below, so the
    # three slices of a head (q, k, v) and its output rows all see one draw;
    # thresholded evaluation reads the logits directly and draws nothing.
    drawing = (units.draw(lambda u: gumbel_sigmoid(u, temperature, noise=noise))
               if units is not None and gates is not None and not deterministic
               else no_intervention())
    with drawing:
        for name, logits in (gates or {}).items():
            # Training samples; evaluation thresholds. Scoring a sampled mask
            # measures a different random subnetwork on every forward pass, and
            # it is not the mask that was learned: a first version sampled
            # everywhere and reported train accuracy flat at 0.70 from 55% of
            # the weights down to 0.02%, because the number never depended on
            # the gates at all.
            sampled = ((logits > 0).to(logits.dtype) if deterministic
                       else gumbel_sigmoid(logits, temperature, noise=noise))
            if units is not None:
                sampled = sampled * units.factor(name, sampled.ndim, deterministic).to(sampled.dtype)
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
    with adapter.edge_gate(edges) if edges else no_intervention():
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

def faith_term(pairs: torch.Tensor, chunk: Sequence[int], faith: Faithfulness,
               reference: Optional[Dict[int, torch.Tensor]]) -> torch.Tensor:
    """The faithfulness term of one batch, by kind

    The kind is an object rather than a string, so what it measures travels
    with the number it produces (faith.py). Which reference it wants is the
    kind's own business: everything that compares against the full model needs
    the row's stored label, and `pair` needs none.
    """
    labels = torch.cat([reference[row] for row in chunk], dim=0) if faith.whole else None
    return faith(pairs, labels)

def ranking_accuracy(pairs: torch.Tensor) -> float:
    return float((pairs[:, 0] > pairs[:, 1]).float().mean())

def first_token_accuracy(adapter, task: CircuitTask, rows: Sequence[int], chunk: int, **kwargs) -> float:
    """Fraction of rows whose argmax over the *whole* vocabulary is the answer

    `ranking_accuracy` asks whether the answer beats one distractor, which two logits
    can satisfy while everything else in the distribution outranks both. This
    asks the question generation actually asks. Chunked because `whole=True`
    returns a row per vocabulary entry and the 1.7B's vocabulary is 151k wide.
    """
    io_all = task.readout(adapter).positive
    right = 0
    for start in range(0, len(rows), chunk):
        block = list(rows)[start:start + chunk]
        final = logit_pairs(adapter, task, block, whole=True, **kwargs)
        want = torch.tensor([io_all[row] for row in block], device=final.device)
        right += int((final.argmax(dim=-1) == want).sum())
    return right / max(1, len(rows))

def split_rows(prompts: Sequence[str], holdout: float) -> "tuple[List[int], List[int]]":
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
        raise SheafError(
            f"{len(prompts)} rows carry only {len(groups)} distinct prompts, which leaves no "
            f"holdout at {holdout:.0%} once rows sharing a prompt are kept together. A mask "
            f"scored on prompts it trained on reports memorization as faithfulness -- widen the "
            f"task's pool rather than its row count, because repeating a prompt adds rows and "
            f"no information."
        )
    train = [row for group in groups[:kept] for row in group]
    test = [row for group in groups[kept:] for row in group]
    return sorted(train), sorted(test)

def chunk_rows(train_rows: Sequence[int], step: int, batch: int) -> List[int]:
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

def attribution(adapter, task: CircuitTask, rows: Sequence[int],
                originals: Dict[str, torch.Tensor], faith: Faithfulness,
                reference: Optional[Dict[int, torch.Tensor]], temperature: float,
                batch: int, batches: int) -> Dict[str, torch.Tensor]:
    """|w * dL/dw| of the faith term on the full model, summed over `batches` batches

    First-order attribution per weight -- what edge attribution patching
    (Syed et al. 2023) computes per edge -- on the unmasked model, so the
    scores say which weights the task's loss is sensitive to before a gate
    has moved. Whole batches of the training rows, the same ones the gates
    will train on; nothing is held out because nothing is fitted.
    """
    leaves = {name: weight.detach().clone().requires_grad_(True)
              for name, weight in originals.items()}
    scores = {name: torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
              for name, weight in originals.items()}
    for index in range(batches):
        chunk = chunk_rows(rows, index, batch)
        pairs = logit_pairs(adapter, task, chunk, None, originals, temperature, deterministic=True,
                       whole=faith.whole, weights=leaves)
        faith_term(pairs, chunk, faith, reference).backward()
        with torch.no_grad():
            for name, leaf in leaves.items():
                if leaf.grad is not None:
                    scores[name] += (leaf.grad * leaf).abs().float()
                    leaf.grad = None
        del pairs
    return scores
