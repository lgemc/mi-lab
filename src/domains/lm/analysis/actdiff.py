"""The activation-difference lens: what a fine-tune adds to the residual stream on average, read as words

The cheapest residual-stream diff there is, and the baseline any dictionary has
to beat. Minder et al. (2025, "Narrow finetuning leaves clearly readable traces
in activation differences") report that the mean difference between a
fine-tune's and its base's residual stream, on text unrelated to the fine-tune
and at the first few positions, already names what the fine-tune was about when
read through the logit lens. It needs no training, and unlike a transcoder it
sees attention: it reads the stream after the whole block.

Per layer: the mean of `h_post - h_pre` over the same positions of the same
sequences, its norm against the base stream's mean norm, and the vocabulary it
promotes and suppresses when pushed through the post checkpoint's final norm
and unembedding. Two windows, because the method's claim is about the early
one: the first few positions after position 0, and every position.

Read the vocabulary as a hypothesis about what changed, not as a result: the
logit lens of a difference vector is a projection, and a direction that the
unembedding reads as noise can still carry what a later layer uses.

A common pipe could be: mean_difference(pre, post, sequences) | top_vocabulary | cosine across fine-tunes
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch

from ..backend.layout import _blocks, _final_norm

# Positions 1..EARLY form the early window; position 0 is the attention sink.
EARLY = 5


def capture_stream(model, ids: torch.Tensor) -> torch.Tensor:
    """Every block's output on one sequence, [layers, positions, width]"""
    seen: Dict[int, torch.Tensor] = {}
    handles = []
    for index, block in enumerate(_blocks(model)):
        def keep(_module, _inputs, out, at=index):
            hidden = out[0] if isinstance(out, tuple) else out
            seen[at] = hidden[0].detach().float()
        handles.append(block.register_forward_hook(keep))
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack([seen[i] for i in sorted(seen)])


@dataclass
class ActDiff:
    """Summed differences and norms per window, ready to be averaged"""
    text_set: str
    sums: Dict[str, torch.Tensor] = field(default_factory=dict)       # window -> [layer, width]
    norms: Dict[str, torch.Tensor] = field(default_factory=dict)      # window -> [layer]
    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, window: str, delta: torch.Tensor, base: torch.Tensor) -> None:
        if delta.shape[1] == 0:
            return
        self.sums[window] = self.sums.get(window, 0) + delta.sum(1)
        self.norms[window] = self.norms.get(window, 0) + base.norm(dim=-1).sum(1)
        self.counts[window] = self.counts.get(window, 0) + delta.shape[1]

    def mean(self, window: str) -> torch.Tensor:
        return self.sums[window] / self.counts[window]

    def relative_norm(self, window: str) -> List[float]:
        """|mean difference| over the base stream's mean norm, per layer"""
        return (self.mean(window).norm(dim=-1) / (self.norms[window] / self.counts[window])).tolist()


def mean_difference(pre, post, sequences: Sequence[Tuple[torch.Tensor, int]], text_set: str) -> ActDiff:
    """Mean residual-stream difference over identical sequences, in the early window and over all scored positions"""
    diff = ActDiff(text_set=text_set)
    for ids, start in sequences:
        h_pre, h_post = capture_stream(pre, ids), capture_stream(post, ids)
        delta = h_post - h_pre
        diff.add("early", delta[:, 1 : 1 + EARLY], h_pre[:, 1 : 1 + EARLY])
        diff.add("all", delta[:, max(start, 1) :], h_pre[:, max(start, 1) :])
    return diff


def top_vocabulary(model, tokenizer, vector: torch.Tensor, k: int = 15) -> Dict[str, List[str]]:
    """What a residual-stream direction promotes and suppresses under the logit lens of `model`"""
    with torch.no_grad():
        normed = _final_norm(model)(vector.to(model.dtype).view(1, 1, -1))
        logits = model.get_output_embeddings()(normed)[0, 0].float()
    up = logits.topk(k).indices.tolist()
    down = (-logits).topk(k).indices.tolist()
    return {"promoted": [tokenizer.decode([i]) for i in up], "suppressed": [tokenizer.decode([i]) for i in down]}
