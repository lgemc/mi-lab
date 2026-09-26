"""Forward passes that exist to be observed: batch, pad, run, and hand the mask over.

Everything a mean ablation or a neuron scan measures is read off a forward
hook while the model runs over a corpus. The adapter's `capture` reads the
residual stream at a position and returns it; these measurements read
somewhere else -- the input to an output projection, the input to an MLP's
down projection -- and want the whole sequence with its padding mask, so they
can sum over real tokens and nothing else. That loop was written four times
across the phase scripts, each with its own truncation length and its own
`padding_side` assignment, and the copies had begun to disagree.

So the loop is here once. `forward_batches` walks the texts in the adapter's
batch size, right-pads, runs the model under `no_grad`, and yields the ids and
mask *after* each pass, which is the moment any hook registered on the model
has just filled whatever it was pointed at. `hooked` is the register/remove
pair around it, written so that a pass that raises still removes its hooks:
a hook left behind is a hook that fires inside the next experiment.

A measurement that needs gradients at those sites rather than values --
attribution -- does not get a loop here, because its loop is not this one: it
runs the same batch several times along an interpolation path and owns what
it writes into the sites each time. What it shares with the forward loop is
`encode`, so the two are looking at the same tokenization, and `Span` /
`scored_positions`, which say which positions a teacher-forced measurement is
about. `differentiable` is here for the same reason: a checkpoint loaded for
inference carries `requires_grad=False` on every weight, and then nothing
downstream is in a graph and every gradient reads as a zero rather than as
the configuration error it is.

Right padding rather than the adapter's left padding for generation, because
a hook summing over positions does not care where the padding sits and right
padding keeps position `i` meaning token `i` for every row -- which is what a
per-token trace wants to print.

A common pipe could be: hooked | forward_batches | mask-weighted sum | mean
A common pipe could be: teacher_forced | differentiable | hooked | activation x gradient
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import torch

from ..core.config import ConfigError

# Long enough for any few-shot translation prompt in the study, short enough
# that a whole-stack capture of every layer's activations still fits beside
# the model on one accelerator.
DEFAULT_MAX_LENGTH = 512

@contextmanager
def hooked(handles: Sequence[torch.utils.hooks.RemovableHandle]) -> Iterator[None]:
    """Remove every handle on exit, whether the block finished or raised"""
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()

def forward_batches(adapter, texts: Sequence[str], batch_size: Optional[int] = None,
                    max_length: int = DEFAULT_MAX_LENGTH,
                    on_batch: Optional[Callable[[torch.Tensor], None]] = None,
                    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Run the model over `texts` and yield (input_ids, attention_mask) after each pass

    `on_batch` is called with the mask before the pass, for a hook that needs
    it while the model is running rather than after.
    """
    size = batch_size or adapter.cfg.batch_size
    for start in range(0, len(texts), size):
        batch = list(texts[start : start + size])
        ids, mask = encode(adapter, batch, max_length)
        if on_batch is not None:
            on_batch(mask)
        with torch.no_grad():
            adapter.model(ids, attention_mask=mask, use_cache=False)
        yield ids, mask

def encode(adapter, batch: Sequence[str], max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """One batch of texts as right-padded (ids, mask) on the model's device"""
    adapter.tokenizer.padding_side = "right"
    encoded = adapter.tokenizer(list(batch), return_tensors="pt", padding=True, truncation=True,
                                max_length=max_length)
    return (encoded["input_ids"].to(adapter.model.device),
            encoded["attention_mask"].to(adapter.model.device))

@dataclass(frozen=True)
class Span:
    """One teacher-forced example: the text the model reads, and where the scored part starts

    `start` is a *token* index into the encoding of `text`: the number of
    tokens the prompt occupies, so positions `start..` are the continuation
    whose likelihood is the measurement. A character offset would be the
    obvious field and the wrong one -- the model is scored per token and the
    boundary has to be one the tokenizer agrees to.

    `teacher_forced` is the only thing that should build these, because the
    boundary is exactly where this goes wrong: BPE merges across it, and
    `tokens(prompt) + tokens(target) != tokens(prompt + target)` whenever it
    does. That function checks the prefix property and refuses the pair
    rather than scoring a continuation that starts one token off.
    """
    text: str
    start: int

def teacher_forced(adapter, prompts: Sequence[str], targets: Sequence[str],
                   joiner: str = " ") -> List[Span]:
    """Prompts and their continuations as spans, with the token boundary checked rather than assumed

    The check is the point. `start` is len(tokenize(prompt)), which is only
    the right boundary if tokenizing the concatenation reproduces the
    prompt's tokens unchanged at the front. Where a merge crosses the join it
    does not, and the scored span would begin inside a token that is half
    prompt -- a quiet off-by-one that shows up as an attribution score for
    the wrong positions and nothing that looks like an error.
    """
    if len(prompts) != len(targets):
        raise ConfigError(f"{len(prompts)} prompts against {len(targets)} targets; teacher forcing needs a pair")
    spans = []
    for index, (prompt, target) in enumerate(zip(prompts, targets, strict=True)):
        head = adapter.tokenizer(prompt)["input_ids"]
        whole = adapter.tokenizer(prompt + joiner + target)["input_ids"]
        if whole[: len(head)] != head:
            raise ConfigError(
                f"tokenizing prompt {index} together with its target does not keep the prompt's "
                f"{len(head)} tokens intact (a merge crosses the join); score this pair with a "
                "joiner the tokenizer does not merge across, such as a newline"
            )
        if len(whole) <= len(head):
            raise ConfigError(f"target {index} is empty once tokenized; there is nothing to score")
        spans.append(Span(text=prompt + joiner + target, start=len(head)))
    return spans

def scored_positions(spans: Sequence[Span], mask: torch.Tensor) -> torch.Tensor:
    """A [batch, seq] float mask that is 1 on the tokens a span is scored over

    Padding is excluded by `mask`, and the prompt by each span's `start`. Both
    are needed: a pass that scores the prompt is measuring the few-shot
    examples, which every prompt in the study shares and no component is
    responsible for.
    """
    positions = torch.arange(mask.shape[1], device=mask.device)[None, :]
    starts = torch.tensor([span.start for span in spans], device=mask.device)[:, None]
    return ((positions >= starts) & mask.bool()).float()

@contextmanager
def differentiable(adapter) -> Iterator[None]:
    """Make the forward pass differentiable without asking for the gradient of a single weight

    A gradient measurement at an activation needs that activation to be in a
    graph, and the obvious way to arrange it -- turn `requires_grad` back on
    for every parameter -- asks autograd for something nobody reads: a
    gradient buffer the size of the model, which on the 8B is sixteen
    gigabytes spent to get at activations that are a thousandth of that.

    All that is actually required is one leaf. Everything downstream of a
    tensor that requires grad requires grad too, so detaching what the first
    block is handed and marking *that* as the leaf puts the whole stack in a
    graph at the cost of one activation. The weights stay frozen, no
    parameter gradient is ever allocated, and the leaf itself is thrown away:
    it exists to be the root of the graph, not to be read.

    The embedding is therefore not attributable through this context, which
    is correct here -- it has no counterfactual mean and nothing ablates it.
    """
    def pre_hook(module, args):
        if not args:
            return None
        return (args[0].detach().requires_grad_(True), *args[1:])

    handle = adapter.blocks[0].register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()

def token_strings(adapter, ids: torch.Tensor, mask: torch.Tensor, row: int) -> List[str]:
    """The real tokens of one padded row, decoded one at a time so positions line up with a trace"""
    real = int(mask[row].sum())
    return [adapter.tokenizer.decode([int(token)]) for token in ids[row, :real]]

def module_owning(adapter, layer: int, target: torch.nn.Module) -> torch.nn.Module:
    """The submodule of block `layer` whose direct children include `target`

    Located rather than named, because the name differs by architecture and
    this file is not the place that knows about any of them -- the backend
    already resolved the projection, so the module holding it is the attention.
    """
    for module in adapter.blocks[layer].modules():
        if any(child is target for child in module.children()):
            return module
    raise ConfigError(f"no module in block {layer} owns {type(target).__name__}")

def attention_of(adapter, layer: int) -> torch.nn.Module:
    """The attention submodule of a block, found by which module owns its output projection"""
    return module_owning(adapter, layer, adapter.projections[layer])
