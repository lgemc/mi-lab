from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List, Optional, Protocol, Sequence, runtime_checkable

import torch

from .config import ConfigError, ModelConfig, Position, load_config

"""
Adapter is the one interface every experiment in this framework is written
against, and the only place a model library is imported. A probe, a patch or a
steering study asks an adapter to capture activations and to generate text; it
never learns whether the activations came out of HuggingFace hooks,
TransformerLens or nnsight-over-vLLM.

Backends are swappable, semantics are not: two backends holding the same
checkpoint must return the same shapes with the same meaning. Only d_model
differs across *models*, never across backends of one model.

The default 'transformers' backend reads the residual stream off the decoder
blocks with forward hooks. It works on every causal LM on the hub, which makes
it the correctness oracle the faster backends get checked against.

It deliberately does *not* use output_hidden_states, which looks like the
obvious way to do this and is subtly not the same thing: that tuple's last
entry is the final layer norm applied to the residual stream, not the residual
stream, so capturing at depth 1.0 would silently hand back a different
quantity than capturing anywhere else. Transformers 5 also implements
output_hidden_states with its own recorder hooks, which can observe a block's
output before an intervention hook has finished with it.

A common pipe could be: load_adapter | capture | steer | generate
"""

BACKENDS: Dict[str, Callable[[ModelConfig], "ModelAdapter"]] = {}

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

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

def register_backend(name: str) -> Callable[[Callable[[ModelConfig], ModelAdapter]], Callable[[ModelConfig], ModelAdapter]]:
    """Register a factory under a backend name, so configs can name it as data"""
    def decorate(factory):
        BACKENDS[name] = factory
        return factory
    return decorate

def _blocks(model) -> torch.nn.ModuleList:
    """Find the list of decoder blocks, whatever this architecture calls it

    Every hook site in this backend is a block boundary, so this is the only
    piece of architecture knowledge the framework needs.
    """
    for path in ("transformer.h", "model.layers", "gpt_neox.layers", "model.decoder.layers"):
        module = model
        for attribute in path.split("."):
            module = getattr(module, attribute, None)
            if module is None:
                break
        if module is not None:
            return module
    raise ConfigError(f"cannot find the decoder blocks of {type(model).__name__}; teach _blocks its layout")

class TransformersAdapter:
    """A ModelAdapter over a HuggingFace causal LM

    Activations are read from output_hidden_states, where entry i + 1 is the
    output of block i -- the residual stream after that block, the site both
    production monitoring setups hook.
    """

    def __init__(self, cfg: ModelConfig, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.blocks = _blocks(model)
        self.cfg = cfg.with_sizes(n_layers=len(self.blocks), d_model=model.config.hidden_size)

    def layer(self, frac: Optional[float] = None) -> int:
        """Resolve a depth fraction to an absolute layer index"""
        return self.cfg.layer(frac)

    def _resolve_layers(self, layers: Optional[Sequence[int]]) -> List[int]:
        """Default to the config's probe layer, and reject indices this model does not have"""
        if layers is None:
            return [self.layer()]
        layers = list(layers)
        if not layers:
            raise ConfigError("capture needs at least one layer")
        out_of_range = [index for index in layers if not 0 <= index < self.cfg.n_layers]
        if out_of_range:
            raise ConfigError(f"layers {out_of_range} are outside the {self.cfg.n_layers} layers of '{self.cfg.id}'")
        return layers

    @contextmanager
    def _record(self, layers: Sequence[int]) -> Iterator[Dict[int, torch.Tensor]]:
        """Collect the residual stream leaving each requested block

        Recording hooks are registered after any steering hook, and PyTorch
        runs forward hooks in registration order while threading the modified
        output through them, so capturing at a steered layer sees the
        intervention rather than the value it replaced.
        """
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(index: int):
            def hook(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[index] = hidden.detach()
            return hook

        for index in layers:
            handles.append(self.blocks[index].register_forward_hook(make_hook(index)))
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def _encode(self, prompts: Sequence[str], padding_side: str):
        """Tokenize every prompt to one uniform length, so chunks stay comparable"""
        self.tokenizer.padding_side = padding_side
        encoded = self.tokenizer(list(prompts), return_tensors="pt", padding=True)
        return encoded["input_ids"], encoded["attention_mask"]

    def capture(
        self,
        prompts: Sequence[str],
        layers: Optional[Sequence[int]] = None,
        position: Position = Position.LAST,
    ) -> torch.Tensor:
        """Capture the residual stream at the given layers, batched per cfg.batch_size

        Chunks are accumulated on CPU: on a unified-memory machine the device
        and host pools are the same silicon, but on a discrete GPU holding a
        thousand prompts' activations resident is how capture OOMs.
        """
        if not prompts:
            raise ConfigError("capture needs at least one prompt")
        position = Position(position)
        layers = self._resolve_layers(layers)
        input_ids, attention_mask = self._encode(prompts, padding_side="right")

        chunks = []
        for start in range(0, len(input_ids), self.cfg.batch_size):
            ids = input_ids[start : start + self.cfg.batch_size].to(self.model.device)
            mask = attention_mask[start : start + self.cfg.batch_size].to(self.model.device)
            with self._record(layers) as captured, torch.no_grad():
                self.model(ids, attention_mask=mask, use_cache=False)
            stacked = torch.stack([captured[index] for index in layers], dim=1)
            chunks.append(_reduce(stacked, mask, position).float().cpu())
        return torch.cat(chunks, dim=0)

    def generate(self, prompts: Sequence[str], max_new_tokens: Optional[int] = None, **kwargs) -> List[str]:
        """Greedily continue each prompt, returning only the new text"""
        input_ids, attention_mask = self._encode(prompts, padding_side="left")
        completions = []
        for start in range(0, len(input_ids), self.cfg.batch_size):
            ids = input_ids[start : start + self.cfg.batch_size].to(self.model.device)
            mask = attention_mask[start : start + self.cfg.batch_size].to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(
                    ids,
                    attention_mask=mask,
                    max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs,
                )
            completions.extend(self.tokenizer.batch_decode(generated[:, ids.shape[1] :], skip_special_tokens=True))
        return completions

    @contextmanager
    def steer(self, layer: int, vector: torch.Tensor, strength: float) -> Iterator[None]:
        """Add a direction to the residual stream leaving `layer`, for the duration of the block

        The vector is normalized and rescaled to the mean activation norm seen
        at that layer in this very forward pass, so a strength of 1.0 means the
        same intervention size on GPT-2 small as on a 27B. At strength 0 no hook
        fires at all, which keeps a zero-strength run byte-identical to no
        steering -- the check that tells you the hook is where you think it is.
        """
        (layer,) = self._resolve_layers([layer])
        if strength == 0:
            yield
            return

        direction = vector.to(self.model.device, self.model.dtype)
        direction = direction / direction.norm()

        def hook(module, args, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            scale = strength * hidden.norm(dim=-1, keepdim=True).mean()
            steered = hidden + scale * direction
            return (steered, *output[1:]) if is_tuple else steered

        handle = self.blocks[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

def _reduce(stacked: torch.Tensor, mask: torch.Tensor, position: Position) -> torch.Tensor:
    """Collapse a [batch, layer, seq, d_model] capture to the requested position

    Padding is never averaged into a MEAN and never mistaken for the LAST
    token: with right padding the final real token is at mask.sum() - 1, which
    is not the final column whenever a prompt is shorter than its batch.
    """
    if position is Position.ALL:
        return stacked
    if position is Position.MEAN:
        weights = mask[:, None, :, None].to(stacked.dtype)
        return (stacked * weights).sum(dim=2) / weights.sum(dim=2)
    last = (mask.sum(dim=1) - 1)[:, None, None, None].expand(-1, stacked.shape[1], -1, stacked.shape[3])
    return stacked.gather(2, last).squeeze(2)

@register_backend("transformers")
def _build_transformers(cfg: ModelConfig) -> ModelAdapter:
    """Load a HuggingFace causal LM in eval mode, with a pad token guaranteed"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_name)
    if tokenizer.pad_token is None:
        # GPT-2 and friends ship no pad token, and batching needs one
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_name, dtype=_DTYPES[cfg.dtype])
    model.to(cfg.device)
    model.eval()
    return TransformersAdapter(cfg, model, tokenizer)

def load_adapter(reference) -> ModelAdapter:
    """Build the adapter a config asks for, from a preset name, a file path or a ModelConfig"""
    cfg = reference if isinstance(reference, ModelConfig) else load_config(reference)
    if cfg.backend not in BACKENDS:
        raise ConfigError(
            f"config '{cfg.id}' asks for backend '{cfg.backend}', which is not registered; "
            f"available backends are {sorted(BACKENDS)}"
        )
    if cfg.dtype not in _DTYPES:
        raise ConfigError(f"unknown dtype '{cfg.dtype}'; known dtypes are {sorted(_DTYPES)}")
    return BACKENDS[cfg.backend](cfg)
