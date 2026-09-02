"""Language-specific neurons the way Tang et al. (2402.16438) find them, and what to do with a flag.

The scan is an activation contrast. Every FFN neuron's mean |activation| is
measured at the input to the MLP's down projection -- the post-activation
vector, which is the one thing in a transformer that is honestly "a neuron"
-- over a target condition and over conditions the target is not: Spanish
translation prompts against bare English and against math and code.
`score = target - mean(others)`, and a neuron more than two sigma above is
flagged. None of that is a causal claim; it says where to look, which is what
phase 1a was for, and the rest of the study is what happens when the flagged
layers are knocked out.

Two things are decided here and not in a script. The neuron vector is read at
the down projection found *by name*, because a hook on the MLP module sees its
output and the neurons are gone by then; and a trace of individual neurons
decodes tokens one at a time from the padded batch, so that position `i` of
the activation is the token printed at position `i` -- the whole point of a
trace being to read what a neuron fires on.

A common pipe could be: control_texts | mean_abs_activation | contrast | flag | top_neurons | trace | summarize
A common pipe could be: earlier top_neurons | contrast | survival
"""

import random
from typing import Any, Dict, List, Sequence, Tuple

import torch

from ..model.passes import DEFAULT_MAX_LENGTH, forward_batches, hooked, token_strings
from .circuits import CircuitError

SIGMAS = 2.0
TOP_FIRING = 12
CONTEXT_TOKENS = 5

# The attribute a decoder MLP stores its down projection under, across the
# families this repo has met. Ordered by how often each is the one.
DOWN_PROJECTIONS = ("down_proj", "c_proj", "dense_4h_to_h", "fc_out", "w2")

class NeuronError(CircuitError):
    """An MLP whose neurons cannot be found, or a contrast that does not line up"""

def control_texts(count: int, seed: int = 0) -> List[str]:
    """Math and code lines with no natural language in them, built locally

    Alternating so that a token budget met early still saw both kinds.
    """
    rng = random.Random(seed)
    lines = []
    for index in range(count):
        if index % 2 == 0:
            a, b, c = rng.randint(2, 999), rng.randint(2, 999), rng.randint(2, 99)
            lines.append(f"{a} * {b} + {c} = {a * b + c}; ({a} + {b}) / {c} = {(a + b) / c:.4f}")
        else:
            name = rng.choice(["x", "acc", "buf", "val", "tmp"])
            lines.append(
                f"def f{index}({name}):\n    return [{name}[i] ** 2 % {rng.randint(3, 97)} "
                f"for i in range(len({name})) if i % {rng.randint(2, 9)} != 0]"
            )
    return lines

def down_projection(mlp: torch.nn.Module, layer: int) -> torch.nn.Module:
    """The linear whose input is the post-activation neuron vector"""
    for name in DOWN_PROJECTIONS:
        module = getattr(mlp, name, None)
        if module is not None:
            return module
    raise NeuronError(f"cannot find the down projection of MLP {layer} ({type(mlp).__name__}); "
                      f"add its attribute name to DOWN_PROJECTIONS")

def mean_abs_activation(adapter, texts: Sequence[str], minimum_tokens: int,
                        max_length: int = DEFAULT_MAX_LENGTH) -> Tuple[torch.Tensor, int]:
    """Mean |activation| per FFN neuron as [n_layers, d_ff] over at least `minimum_tokens` real tokens

    Stops after the batch that meets the budget, so every condition is
    measured over about the same number of tokens rather than about the same
    number of texts -- code lines are longer than sentences.
    """
    downs = [down_projection(mlp, layer) for layer, mlp in enumerate(adapter.mlps)]
    state: Dict[str, Any] = {}
    sums = None
    tokens = 0

    def make_hook(layer: int):
        def hook(module, args):
            value = args[0].detach()
            state["last"][layer] = (value.abs().float() * state["mask"][..., None]).sum(dim=(0, 1)).cpu()
        return hook

    def on_batch(mask):
        state["mask"] = mask.float()
        state["last"] = {}

    handles = [down.register_forward_pre_hook(make_hook(layer)) for layer, down in enumerate(downs)]
    with hooked(handles):
        for _, mask in forward_batches(adapter, texts, max_length=max_length, on_batch=on_batch):
            stacked = torch.stack([state["last"][layer] for layer in range(len(downs))]).double()
            sums = stacked if sums is None else sums + stacked
            tokens += int(mask.sum())
            if tokens >= minimum_tokens:
                break
    if sums is None:
        raise NeuronError("no texts to measure over")
    return sums / tokens, tokens

def contrast(target: torch.Tensor, others: Sequence[torch.Tensor]) -> torch.Tensor:
    """The score: the target condition's mean |activation| minus the mean of the others'"""
    if not others:
        raise NeuronError("a contrast needs at least one condition to contrast against")
    for other in others:
        if other.shape != target.shape:
            raise NeuronError(f"condition shapes differ: {tuple(target.shape)} vs {tuple(other.shape)}")
    return target - torch.stack(list(others)).mean(dim=0)

def flag(score: torch.Tensor, sigmas: float = SIGMAS) -> Tuple[torch.Tensor, float]:
    """(mask of neurons more than `sigmas` standard deviations above zero, sigma)"""
    sigma = float(score.std())
    return score > sigmas * sigma, sigma

def top_neurons(score: torch.Tensor, count: int) -> List[Dict[str, Any]]:
    """The highest-scoring neurons as {layer, neuron, score}, best first"""
    d_ff = score.shape[1]
    values, order = score.flatten().topk(min(count, score.numel()))
    return [{"layer": int(index // d_ff), "neuron": int(index % d_ff), "score": round(float(value), 6)}
            for value, index in zip(values, order, strict=True)]

def concentration(flagged: torch.Tensor) -> Dict[str, Any]:
    """How the flagged neurons split between the bottom quarter, the top quarter and the middle of the stack

    Tang et al.'s finding is that language neurons sit at both ends; this is
    the number that says whether this model does the same.
    """
    n_layers = flagged.shape[0]
    quarter = max(1, n_layers // 4)
    per_layer = flagged.sum(dim=1)
    total = int(flagged.sum())
    bottom = int(per_layer[:quarter].sum())
    top = int(per_layer[-quarter:].sum())
    return {
        "bottom_quarter_layers": quarter,
        "bottom_quarter_flagged": bottom,
        "top_quarter_flagged": top,
        "middle_flagged": total - bottom - top,
        "bottom_plus_top_share": round((bottom + top) / total, 4) if total else None,
    }

def per_layer_counts(flagged: torch.Tensor) -> Dict[str, int]:
    """{layer: flagged count} for every layer with at least one, keyed as strings for JSON"""
    return {str(layer): int(count) for layer, count in enumerate(flagged.sum(dim=1)) if int(count)}

def trace(adapter, texts: Sequence[str], layer: int, neurons: Sequence[int],
          max_length: int = DEFAULT_MAX_LENGTH) -> List[Tuple[torch.Tensor, List[str]]]:
    """Per-token activations of the chosen neurons: [(activations [seq, n], token strings)] per text"""
    down = down_projection(adapter.mlps[layer], layer)
    grabbed: Dict[str, torch.Tensor] = {}
    columns = list(neurons)

    def hook(module, args):
        grabbed["value"] = args[0].detach()[..., columns].float().cpu()

    traces = []
    with hooked([down.register_forward_pre_hook(hook)]):
        for ids, mask in forward_batches(adapter, texts, max_length=max_length):
            for row in range(ids.shape[0]):
                tokens = token_strings(adapter, ids, mask, row)
                traces.append((grabbed["value"][row, :len(tokens)], tokens))
    return traces

def summarize(traces: Sequence[Tuple[torch.Tensor, List[str]]], column: int, top: int = TOP_FIRING,
              sigmas: float = SIGMAS) -> Dict[str, Any]:
    """One neuron's behaviour over a condition: moments plus its hardest-firing tokens in context"""
    if not traces:
        raise NeuronError("no traces to summarize")
    values = torch.cat([activations[:, column] for activations, _ in traces])
    firing = []
    for activations, tokens in traces:
        for position in range(len(tokens)):
            firing.append((float(activations[position, column]), position, tokens))
    firing.sort(key=lambda item: -abs(item[0]))
    hottest = [
        {
            "activation": round(value, 2),
            "token": tokens[position],
            "context": "".join(tokens[max(0, position - CONTEXT_TOKENS) : position + 1]),
        }
        for value, position, tokens in firing[:top]
    ]
    magnitude = values.abs()
    above = float((magnitude > magnitude.mean() + sigmas * magnitude.std()).float().mean())
    return {
        "mean_abs": round(float(magnitude.mean()), 3),
        "max_abs": round(float(magnitude.max()), 2),
        "share_tokens_above_2sigma": round(above, 4),
        "top_firing": hottest,
    }

def survival(earlier: Sequence[Dict[str, Any]], score: torch.Tensor, count: int,
             sigmas: float = SIGMAS) -> List[Dict[str, Any]]:
    """How an earlier scan's top neurons fare under a second contrast

    Each entry keeps its earlier fields and gains the new score (raw and in
    sigmas), whether it is in the new top `count`, and whether it is flagged
    at all. A neuron that was top under the translation prompt and vanishes
    under raw Spanish belongs to the task framing, not the language.
    """
    sigma = float(score.std())
    top_keys = {(entry["layer"], entry["neuron"]) for entry in top_neurons(score, count)}
    survivors = []
    for entry in earlier:
        key = (entry["layer"], entry["neuron"])
        value = float(score[key[0], key[1]])
        survivors.append({
            **entry,
            "second_score": round(value, 6),
            "second_sigmas": round(value / sigma, 2) if sigma else None,
            "in_second_top": key in top_keys,
            "flagged_second": bool(value > sigmas * sigma),
        })
    return survivors
