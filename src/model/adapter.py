from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Protocol, Sequence, runtime_checkable

import torch

from ..core.config import ConfigError, ModelConfig, Position, load_config

"""
Adapter is the one interface every experiment in this framework is written
against. A probe, a patch or a steering study asks an adapter to capture
activations and to generate text; it never learns whether the activations came
out of HuggingFace hooks, TransformerLens or nnsight-over-vLLM.

This module is the contract and nothing else. No model library is imported
here and no architecture is named -- that lives in backends/, one module per
implementation, each registering itself under the string key a config names.
The split is what the registry is for: `configs/qwen3.5-27b.yaml` asks for a
backend that does not exist yet, and adding it should be a new file next to
the existing one rather than an edit to this one.

Backends are swappable, semantics are not: two backends holding the same
checkpoint must return the same shapes with the same meaning. Only d_model
differs across *models*, never across backends of one model. Those shared
semantics are the whole reason this file states the protocols separately from
any code that satisfies them.

Circuit work needs more than the residual stream -- it needs the heads that
wrote into it, the attention that routed them, and the ability to overwrite
either from another run. That is a second protocol, CircuitAdapter, rather
than a second import: a backend serving batched generations behind an API can
honestly implement capture and steer and honestly cannot implement head
patching, and the difference should be a failed isinstance check with a
message, not an AttributeError halfway through an experiment.

A common pipe could be: load_adapter | capture | steer | generate
"""

BACKENDS: Dict[str, Callable[[ModelConfig], "ModelAdapter"]] = {}

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

def resolve_device(name: str = "auto") -> str:
    """Turn a config's device into one this machine actually has

    "auto" is the default and it means the best accelerator present, falling
    back to the CPU. It lives here rather than in core/config.py because the
    answer is a fact about the machine and needs torch to ask, and config.py
    imports nothing -- a config has to stay readable where there is no CUDA,
    no GPU and no model library at all.

    Anything else is passed through untouched and is therefore a demand: a
    config that says "cuda" on a machine without one should fail loudly when
    the weights are placed, because it was written by somebody who knew what
    they wanted. Only "auto" is allowed to settle for less.
    """
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@runtime_checkable
class ModelAdapter(Protocol):
    """One interface, any backend: hooks and generation with no model facts baked in"""

    cfg: ModelConfig

    def layer(self, frac: Optional[float] = None) -> int:
        """Resolve a depth fraction to an absolute layer index"""

    def capture(
        self,
        prompts: Sequence[str],
        layers: Optional[Sequence[int]] = None,
        position: Position = Position.LAST,
    ) -> torch.Tensor:
        """Residual stream activations as [batch, layer, d_model], or [batch, layer, seq, d_model] for ALL"""

    def generate(self, prompts: Sequence[str], max_new_tokens: Optional[int] = None, **kwargs) -> List[str]:
        """Greedy continuations of each prompt, prompt text stripped off"""

    @contextmanager
    def steer(self, layer: int, vector: torch.Tensor, strength: float) -> Iterator[None]:
        """Add strength * vector to the residual stream at `layer` for the duration of the block"""

@dataclass(frozen=True)
class Unembedding:
    """The map from one residual stream write to the logit difference it is directly worth

    The final norm is not linear, so a single component's write cannot be
    turned into logits on its own without a convention. The convention here is
    the usual one: freeze the normalizer's divisor at the value the *complete*
    residual stream produced on this forward pass. What is left of the norm is
    then affine, and the decomposition is exactly additive again.

    Freezing is an approximation, and it is the interesting one. It says: had
    this component not written, everything else -- including how hard the final
    norm divides -- would have stayed as it was. That is false in detail, and
    it is precisely what makes a direct attribution a claim about the direct
    path and nothing else.

    Only differences of two logits are offered, never a logit on its own. A
    single logit carries the softmax's arbitrary additive constant, and every
    component appears to move it; the difference between two tokens is the
    quantity the behaviour is actually made of.
    """
    weight: torch.Tensor
    gain: torch.Tensor
    bias: torch.Tensor
    scale: torch.Tensor
    centers: bool

    def offset(self, positive: Sequence[int], negative: Sequence[int]) -> torch.Tensor:
        """The part of the logit difference that belongs to no component at all

        The final norm's own shift is added after the residual stream has been
        read, so it lands on the answer without any head having written it.
        Small, constant, and worth keeping visible: it is the difference
        between "the parts sum to the whole" and "the parts nearly sum to the
        whole", and chasing that gap through a head-level attribution is a
        wasted afternoon.
        """
        delta = (self.weight[list(positive)] - self.weight[list(negative)]).float()
        return (self.bias * delta).sum(dim=-1)

    def logit_difference(
        self, contributions: torch.Tensor, positive: Sequence[int], negative: Sequence[int]
    ) -> torch.Tensor:
        """Score residual stream writes as logit(positive) - logit(negative)

        contributions is [batch, ..., d_model] and the answer is [batch, ...],
        so a whole [batch, layer, head, d_model] stack of head writes scores in
        one call. positive and negative are one token id per batch row, because
        every IOI prompt asks about a different pair of names.
        """
        writes = contributions.float()
        if writes.shape[0] != len(positive) or writes.shape[0] != len(negative):
            raise ConfigError(
                f"{writes.shape[0]} contributions but {len(positive)} positive and "
                f"{len(negative)} negative token ids; they index the same batch"
            )
        delta = (self.weight[list(positive)] - self.weight[list(negative)]).float() * self.gain
        if self.centers:
            # a LayerNorm subtracts the mean, so the all-ones direction of a write reaches no logit
            writes = writes - writes.mean(dim=-1, keepdim=True)
        spare = writes.dim() - 2
        delta = delta.reshape(delta.shape[0], *([1] * spare), delta.shape[-1])
        return (writes * delta).sum(dim=-1) / self.scale.reshape(-1, *([1] * spare))

@dataclass(frozen=True)
class Decomposition:
    """Every write into the residual stream at the final token, split by who made it

    The residual stream is a sum, so this is a partition rather than an
    attribution method: embedding, each head, each attention output bias and
    each MLP, adding up to the vector the unembedding actually reads. The
    remainder is kept and reported instead of being folded into a catch-all
    term -- a decomposition whose leftovers are defined as "everything else"
    can never be wrong, and therefore never tells you it is missing a site.
    """
    heads: torch.Tensor
    mlps: torch.Tensor
    biases: torch.Tensor
    embedding: torch.Tensor
    residual: torch.Tensor
    unembedding: Unembedding

    @property
    def total(self) -> torch.Tensor:
        """Everything the split accounts for, summed back up"""
        return self.embedding + self.heads.sum(dim=(1, 2)) + self.mlps.sum(dim=1) + self.biases.sum(dim=1)

    @property
    def remainder(self) -> torch.Tensor:
        """What the split failed to account for: numerically zero when every site is covered"""
        return self.residual - self.total

@runtime_checkable
class CircuitAdapter(ModelAdapter, Protocol):
    """The extra surface circuit work needs: heads, attention, logits and patching

    Everything here addresses a site *inside* a block, which is why it is a
    separate contract. A backend that can only hand back hidden states can
    still be a ModelAdapter and probe fine; it cannot answer which head moved
    the name.
    """

    def logits(self, prompts: Sequence[str]) -> torch.Tensor:
        """Next-token logits at each prompt's final real token, as [batch, vocab]"""

    def attention(self, prompts: Sequence[str], layers: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Attention weights as [batch, layer, head, query, key]"""

    def head_outputs(self, prompts: Sequence[str], layers: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Each head's output before the projection that mixes them, as [batch, layer, head, seq, d_head]"""

    def head_gradients(
        self,
        prompts: Sequence[str],
        positive: Sequence[int],
        negative: Sequence[int],
        layers: Optional[Sequence[int]] = None,
        toward: Optional[Sequence[str]] = None,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """d(logit(positive) - logit(negative)) / d(head output), as [batch, layer, head, seq, d_head]

        The same site head_outputs reads and patch writes, differentiated
        rather than recorded. One backward pass answers for every head at
        once, which is what makes a gradient-based approximation of patching
        cost a constant number of passes instead of one per site.

        The token ids index the batch the way logit_difference's do: one pair
        per prompt, because a two-answer task asks about a different pair in
        every row.
        """

    def decompose(self, prompts: Sequence[str]) -> Decomposition:
        """Split the final token's residual stream into the writes that built it"""

    @contextmanager
    def patch(
        self,
        residual: Optional[Dict[int, torch.Tensor]] = None,
        heads: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    ) -> Iterator[None]:
        """Overwrite activations with values from another run, for the duration of the block"""

    def single_token(self, text: str) -> int:
        """The id of a string that is exactly one token on this model"""

    def tokens(self, prompt: str) -> List[str]:
        """The prompt as the strings the model actually sees, for labelling an axis"""

def require_circuits(adapter: ModelAdapter) -> CircuitAdapter:
    """Assert that this backend exposes the circuit surface, naming it if it does not"""
    if not isinstance(adapter, CircuitAdapter):
        raise ConfigError(
            f"backend '{adapter.cfg.backend}' does not implement the circuit surface "
            "(logits, attention, head_outputs, head_gradients, decompose, patch); "
            "circuit experiments need a backend that does"
        )
    return adapter

BackendFactory = Callable[[ModelConfig], ModelAdapter]

def register_backend(name: str) -> Callable[[BackendFactory], BackendFactory]:
    """Register a factory under a backend name, so configs can name it as data"""
    def decorate(factory):
        BACKENDS[name] = factory
        return factory
    return decorate

def load_adapter(reference) -> ModelAdapter:
    """Build the adapter a config asks for, from a preset name, a file path or a ModelConfig"""
    cfg = reference if isinstance(reference, ModelConfig) else load_config(reference)
    if cfg.backend not in BACKENDS:
        raise ConfigError(
            f"config '{cfg.id}' asks for backend '{cfg.backend}', which is not registered; "
            f"available backends are {sorted(BACKENDS)}"
        )
    if cfg.dtype not in DTYPES:
        raise ConfigError(f"unknown dtype '{cfg.dtype}'; known dtypes are {sorted(DTYPES)}")
    return BACKENDS[cfg.backend](cfg)

# Imported for its side effect, and imported last on purpose. Registration has to
# happen when this module does -- BACKENDS is empty until something fills it, and a
# caller should never have to remember to import a backend by hand -- while the
# backend itself imports the protocols above. Anywhere but the bottom of this file
# and that is a cycle.
from .backends import transformers as _transformers  # noqa: E402, F401
