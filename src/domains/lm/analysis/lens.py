import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch

from ....core.config import Position
from ....core.metrics import degeneracy
from ..backend.layout import _final_norm

"""
The cheapest possible readout: what every layer would say if it were the last.

A logit lens takes the residual stream at layer L, applies the normalization
that sits before the unembedding, and multiplies by the unembedding matrix.
The answer is a distribution over the vocabulary -- one token per layer, per
position -- and it is the floor every other activation-to-text method has to
beat, because it costs one forward pass and no training at all.

What it can and cannot be asked matters here. A lens emits a token, so it can
answer a WorkspaceBench bank whose read position is the final prompt token and
whose intermediate is a word; it cannot answer a bank that wants a sentence
about what the model is planning. Running it on the second kind and reporting
a zero is not a finding, it is a category error, so `single_token_banks` picks
the ones the instrument fits.

The scorers are deliberately three, and they disagree on purpose:

- `hit`, whether any of the top-k tokens at a layer is the intermediate. The
  headline number, and the optimistic one -- a lens that puts the word at rank
  9 of layer 40 gets the same credit as one that puts it at rank 1.
- `leaked`, whether the model's own answer already contains the intermediate.
  The premise of the benchmark is that the model computes the word and never
  writes it; where it writes it anyway, a "successful" readout is reading the
  output, not the workspace.
- `degeneracy`, whether the generations are still language. A lens run against
  a model that is producing mush is measuring the mush.

A Jacobian lens (J-lens; Anthropic, "Verbalizable Representations Form a
Global Workspace in Language Models", 2026) is the same readout with one
matrix in front: the residual at layer L is carried into the final layer's
basis by J_L = E[d h_final / d h_L], averaged over a web-text corpus, and
then decoded exactly as the logit lens decodes. `JacobianLens` reads the
reference implementation's fitted file (`anthropics/jacobian-lens`, published
per model under `neuronpedia/jacobian-lens`) and `transport` is the whole of
the difference, so every method here is scored by the same code on the same
items -- which is what makes a leaderboard of them mean anything.

A common pipe could be: WorkspaceBench | logit_lens(transport=lens.transport) | score_bank | report
"""

# A token this short matching by prefix is a coincidence: 'a' prefixes almost
# every target. Three characters is where a BPE fragment ('Feb' for February)
# starts being evidence rather than noise.
PREFIX_FLOOR = 3

@dataclass(frozen=True)
class LensReadout:
    """What one layer said at one position: the top tokens and their probabilities"""
    layer: int
    tokens: List[str]
    scores: List[float]

@dataclass
class ItemScore:
    """One benchmark item, read at every layer and judged three ways"""
    name: str
    family: str
    intermediates: List[str]
    answer: str = ""
    hit_layers: List[int] = field(default_factory=list)
    best_layer: Optional[int] = None
    best_token: str = ""
    leaked: bool = False

    @property
    def hit(self) -> bool:
        return bool(self.hit_layers)

def normalize(text: str) -> str:
    """Casefold and strip a token down to what two spellings of a word share

    BPE tokens arrive with a leading space or a 'Ġ', targets arrive as words,
    and Portuguese arrives with combining accents that compare unequal to the
    same letters composed. All three are spelling, not content.
    """
    folded = unicodedata.normalize("NFKC", text).replace("Ġ", " ").replace("▁", " ")
    return folded.strip().strip("\"'.,;:!?()[]").casefold()

def matches(token: str, target: str) -> bool:
    """Whether one readout token is the target word

    A lens emits one token and a target can be several, so a token that is a
    prefix of the target counts: 'Feb' is the model having February, not the
    model missing it. The reverse -- a token containing the target -- counts
    too, for the tokenizers that swallow a short word into a longer piece.
    """
    left, right = normalize(token), normalize(target)
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) >= PREFIX_FLOOR and (right.startswith(left) or left.startswith(right)):
        return True
    return len(right) >= PREFIX_FLOOR and right in left

class JacobianLens:
    """A fitted Jacobian lens: one [d_model, d_model] matrix per source layer

    The file is the reference implementation's own `save()` -- `J` keyed by
    layer, plus `n_prompts`, `source_layers` and `d_model` -- read as it is
    written rather than converted, so the lens scored here is the published
    one. The site is the one `capture` reads: the output of block L.

    The last layer has no matrix and needs none: its Jacobian onto itself is
    the identity, and the J-lens there *is* the logit lens. `transport`
    passes it through, so a J-lens run reads every layer a logit-lens run does.

    An R-lens (RelP lens) is the same file with different matrices: the
    Jacobian taken through an LRP-modified backward graph -- RMSNorm scale,
    the SiLU gate's sigmoid and half of the SwiGLU product detached -- with the
    forward pass unchanged. Its rules and fitting recipe (corpus, target layer,
    prompt count) arrive in `provenance` and are recorded with every result.
    """

    def __init__(self, jacobians: Dict[int, torch.Tensor], n_prompts: int = 0, source: str = "",
                 provenance: Optional[Dict] = None):
        if not jacobians:
            raise ValueError("a Jacobian lens needs at least one layer's matrix")
        self.jacobians = jacobians
        self.n_prompts = n_prompts
        self.source = source
        self.provenance = provenance or {}
        self.d_model = next(iter(jacobians.values())).shape[-1]
        self._device: Dict[int, torch.Tensor] = {}

    @classmethod
    def load(cls, path, source: str = "") -> "JacobianLens":
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        if "J" not in state:
            raise ValueError(f"{path} is not a Jacobian lens: it has no 'J' ({sorted(state)})")
        # weights_only: a lens is tensors and plain data, and a file that needs more than that to load
        # is running code from whoever published it. The R-lens pairs load this way too.
        return cls({int(layer): matrix for layer, matrix in state["J"].items()},
                   n_prompts=int(state.get("n_prompts", 0)), source=source or str(path),
                   provenance=dict(state.get("provenance") or {}))

    @property
    def layers(self) -> List[int]:
        return sorted(self.jacobians)

    def check(self, adapter) -> None:
        """Refuse a lens fitted on another model, which would transport into the wrong basis silently"""
        if self.d_model != adapter.cfg.d_model:
            raise ValueError(f"lens is {self.d_model} wide and '{adapter.cfg.id}' is {adapter.cfg.d_model}")
        beyond = [layer for layer in self.layers if layer >= adapter.cfg.n_layers]
        if beyond:
            raise ValueError(f"lens has layers {beyond} and '{adapter.cfg.id}' has {adapter.cfg.n_layers}")

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """h @ J_L^T in float32, as the reference does; the identity where no matrix was fitted"""
        if layer not in self.jacobians:
            return residual
        if layer not in self._device:
            # one layer resident at a time: a 27B's lens is several GB and only one is used per step
            self._device = {layer: self.jacobians[layer].to(residual.device, torch.float32)}
        return residual.float() @ self._device[layer].T

def logit_lens(
    adapter,
    prompts: Sequence[str],
    layers: Optional[Sequence[int]] = None,
    position: Position = Position.LAST,
    top_k: int = 10,
    transport: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
) -> List[List[LensReadout]]:
    """Decode every layer's residual stream through the final norm and unembedding

    The residual comes from `capture`, which reads block outputs rather than
    `output_hidden_states` -- that tuple's last entry is already normalized,
    and a lens that took it would be reading a different quantity at the top
    of the stack than at the bottom.

    Scores are probabilities, not logits. Logits from an early layer are not
    on the same scale as logits from a late one, and a table of both invites
    a comparison that means nothing.

    `transport` maps a layer's residual before it is decoded -- a
    `JacobianLens.transport` makes this a J-lens. It runs in float32 on the
    model's device and the result is cast back, as the reference casts to the
    model's dtype before its final norm.
    """
    if not prompts:
        raise ValueError("a lens needs at least one prompt")
    layers = list(layers if layers is not None else range(adapter.cfg.n_layers))
    residual = adapter.capture(prompts, layers=layers, position=position)  # [batch, layer, d_model], cpu float32

    norm = _final_norm(adapter.model)
    unembedding = adapter.model.get_output_embeddings().weight
    device, dtype = unembedding.device, unembedding.dtype

    readouts: List[List[LensReadout]] = [[] for _ in prompts]
    with torch.no_grad():
        for index, layer in enumerate(layers):
            hidden = residual[:, index].to(device=device)
            if transport is not None:
                hidden = transport(hidden, layer)
            logits = norm(hidden.to(dtype)) @ unembedding.T
            probabilities = torch.softmax(logits.float(), dim=-1)
            scores, ids = probabilities.topk(top_k, dim=-1)
            for row in range(len(prompts)):
                readouts[row].append(LensReadout(
                    layer=layer,
                    tokens=[adapter.tokenizer.decode([int(i)]) for i in ids[row]],
                    scores=[round(float(s), 6) for s in scores[row]],
                ))
    return readouts

def item_name(item: Dict) -> str:
    """The key an item is stored and matched under across runs"""
    return str(item.get("name") or item.get("id") or "?")

def score_item(item: Dict, readouts: Sequence[LensReadout], answer: str, family: str) -> ItemScore:
    """Judge one item's readouts against the intermediates it was built around

    The best layer is the earliest one that hit, not the one with the highest
    probability: the question a lens answers is *where* the model is holding
    the word, and a later layer hitting too is the word surviving, not a
    better read.
    """
    targets = [str(target) for target in item.get("intermediates", [])]
    score = ItemScore(name=item_name(item), family=family, intermediates=targets)
    score.answer = answer
    score.leaked = any(normalize(target) in normalize(answer) for target in targets if target)
    for readout in readouts:
        found = next((token for token in readout.tokens
                      for target in targets if matches(token, target)), None)
        if found is not None:
            score.hit_layers.append(readout.layer)
            if score.best_layer is None:
                score.best_layer, score.best_token = readout.layer, found
    return score

def score_bank(adapter, items: Sequence[Dict], family: str = "?", layers: Optional[Sequence[int]] = None,
               top_k: int = 10, max_new_tokens: Optional[int] = None,
               transport: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
               answers: Optional[Dict[str, str]] = None) -> List[ItemScore]:
    """Run the lens and the model over a bank's items, in batches, and score each one

    Items rather than a WorkspaceBench, because what is scored is always a
    filtered subset -- `readable()` drops what a one-token readout cannot be
    asked -- and a function taking the whole bank would invite scoring the
    items it cannot answer.

    `answers` are the model's own continuations from an earlier run, by item
    name. They depend on the model and the generation settings and not on the
    readout, so a second method on the same model reuses them rather than
    generating again -- which is half the cost of a run. An item missing from
    them is generated as usual.
    """
    items = list(items)
    prompts = [item["prompt"] for item in items]

    scores: List[ItemScore] = []
    size = max(1, adapter.cfg.batch_size)
    for start in range(0, len(items), size):
        chunk, texts = items[start:start + size], prompts[start:start + size]
        readouts = logit_lens(adapter, texts, layers=layers, top_k=top_k, transport=transport)
        names = [item_name(item) for item in chunk]
        if answers is not None and all(name in answers for name in names):
            said = [answers[name] for name in names]
        else:
            said = adapter.generate(texts, max_new_tokens=max_new_tokens)
        scores.extend(score_item(item, rows, answer, family)
                      for item, rows, answer in zip(chunk, readouts, said, strict=True))
    return scores

def report(scores: Sequence[ItemScore]) -> Dict:
    """The three numbers and the layer curve, for one bank or several

    `hit_rate` is over every item; `clean_hit_rate` drops the items whose
    answer leaked the intermediate, and is the number to quote -- on those
    items a readout cannot be distinguished from the model simply having said
    the word.
    """
    if not scores:
        return {"items": 0}
    clean = [score for score in scores if not score.leaked]
    layers = sorted({layer for score in scores for layer in score.hit_layers})
    curve = {str(layer): round(sum(layer in score.hit_layers for score in scores) / len(scores), 4)
             for layer in layers}
    best = [score.best_layer for score in scores if score.best_layer is not None]
    return {
        "items": len(scores),
        "hit_rate": round(sum(score.hit for score in scores) / len(scores), 4),
        "clean_hit_rate": round(sum(score.hit for score in clean) / len(clean), 4) if clean else None,
        "leak_rate": round(sum(score.leaked for score in scores) / len(scores), 4),
        "degeneracy": round(degeneracy([score.answer for score in scores]), 4),
        "median_first_hit_layer": sorted(best)[len(best) // 2] if best else None,
        "hit_rate_by_layer": curve,
    }
