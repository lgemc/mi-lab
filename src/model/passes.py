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

Right padding rather than the adapter's left padding for generation, because
a hook summing over positions does not care where the padding sits and right
padding keeps position `i` meaning token `i` for every row -- which is what a
per-token trace wants to print.

A common pipe could be: hooked | forward_batches | mask-weighted sum | mean
"""

from contextlib import contextmanager
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
        adapter.tokenizer.padding_side = "right"
        encoded = adapter.tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                    max_length=max_length)
        ids = encoded["input_ids"].to(adapter.model.device)
        mask = encoded["attention_mask"].to(adapter.model.device)
        if on_batch is not None:
            on_batch(mask)
        with torch.no_grad():
            adapter.model(ids, attention_mask=mask, use_cache=False)
        yield ids, mask

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
